import csv
import subprocess
import math
import os
import sys
from src.config import *
from src.model import *
from src.type import *


class Ramulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=5,
                 ffn_sparsity=0.95):
        self.df = []
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
        self.hdim = modelinfos['hdim']
        self.ff_dim = int(modelinfos['ff_scale'] * modelinfos['hdim'])
        self.fast_mode = fast_mode
        if ffn_sparsity < 0 or ffn_sparsity >= 1:
            raise ValueError("ffn_sparsity must be in [0, 1).")
        self.ffn_sparsity = ffn_sparsity

    def _log_columns(self):
        return [
            'layer_type', 'M', 'N', 'K', 'nhead', 'dbyte', 'pim_type',
            'power_constraint', 'sparsity', 'cycle', 'mac', 'softmax',
            'mvgb', 'mvsb', 'wrgb'
        ]

    def _load_log(self):
        columns = self._log_columns()
        if self.df:
            return
        if not os.path.exists(self.output_log):
            return
        with open(self.output_log, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or any(
                    col not in reader.fieldnames for col in columns):
                self.df = []
                return
            self.df = [{col: row[col] for col in columns} for row in reader]

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += "      channel: 16\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: HBM3_5.2Gbps\n"
        else:
            line += "      preset: HBM3_5.2Gbps_NPC\n"
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: HBM3-PIM\n"
        with open(yaml_file, 'w') as f:
            f.write(line)

    def update_log_file(self, log):
        self._load_log()
        columns = self._log_columns()
        row = {col: log[i] for i, col in enumerate(columns)}
        row_key = tuple(str(row[col]) for col in columns)
        existing_keys = {tuple(str(r[col]) for col in columns) for r in self.df}
        if row_key not in existing_keys:
            self.df.append(row)
        with open(self.output_log, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(self.df)

    #def run_ramulator(self):
    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name, layer_type):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')
        # 2025-08-07 추가 / select trace gen file using layer type
        if layer_type == LayerType.FFN:
            if pim_type != PIMType.BA:
                raise NotImplementedError(
                    "FFN trace generation is currently implemented only for bank-level PIM."
                )
            layer_type_name = "ff"
            trace_args = "--dmodel {} --ffdim {} --dbyte {} --sparsity {} --output {}".format(
                self.hdim, l, dbyte, self.ffn_sparsity, trace_file)
        else:
            layer_type_name = "attention"
            trace_args = "--dhead {} --nhead {} --seqlen {} --dbyte {} --output {}".format(
                self.dhead, num_ops_per_hbm, l, dbyte, trace_file)

        trace_exc = os.path.join(
            self.ramulator_dir,
            "trace_gen/gen_trace_attacc_{}_{}.py".format(pim_type_name, layer_type_name))
        # trace_args = "--dhead {} --nhead {} --seqlen {} --dbyte {} --output {}".format(
        #     self.dhead, num_ops_per_hbm, l, dbyte, trace_file)

        gen_trace_cmd = f"{sys.executable} {trace_exc} {trace_args}"

        # generate trace
        try:
            subprocess.run(gen_trace_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # run ramulator
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        env = os.environ.copy()
        ramulator_lib_dir = os.path.abspath(self.ramulator_dir)
        if env.get("LD_LIBRARY_PATH"):
            env["LD_LIBRARY_PATH"] = "{}:{}".format(
                ramulator_lib_dir, env["LD_LIBRARY_PATH"])
        else:
            env["LD_LIBRARY_PATH"] = ramulator_lib_dir
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    check=True,
                                    env=env,
                                    shell=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # remove trace
        rm_trace_cmd = f"rm {trace_file}"
        try:
            os.system(rm_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # parsing output
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        out = [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
            n_cmds["wrgb"]
        ]
        return out

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            m, n, k, num_ops, dbyte = layer.get_infos()
            dhead = k if layer.type == LayerType.FFN else self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            lt = 1
            if layer.type == LayerType.FFN:
                lt = 2

            sparsity_tag = int(
                self.ffn_sparsity * 1000) if layer.type == LayerType.FFN else 0
            file_name = "attacc_m{}_n{}_k{}_nattn{}_dbyte{}_pc{}_layer{}_sp{}".format(
                m, n, k, num_ops_per_hbm, layer.dbyte,
                int(power_constraint), lt, sparsity_tag)
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(pim_type, l, num_ops_per_hbm,
                                        layer.dbyte, yaml_file, file_name, layer.type)

            # remove trace
            rm_yaml_cmd = f"rm {yaml_file}"
            try:
                os.system(rm_yaml_cmd)
            except Exception as e:
                print(f"Error: {e}")

            # post processing
            # 32: read granularity
            cycle, mac, sfm, mvgb, mvsb, wrgb = result
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 1

            ## update log file

            log = [
                layer.type.name, m, n, k, num_ops_per_hbm, dbyte,
                pim_type.name, power_constraint,
                self.ffn_sparsity if layer.type == LayerType.FFN else 0
            ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        self._load_log()
        if not self.df:
            self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = layer.numOp
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        m, n, k, num_ops, dbyte = layer.get_infos()
        sparsity = self.ffn_sparsity if layer.type == LayerType.FFN else 0
        dbyte = layer.dbyte
        def bool_value(value):
            if isinstance(value, bool):
                return value
            return str(value).lower() == 'true'

        row = [
            r for r in self.df
            if r['layer_type'] == layer.type.name and int(r['M']) == m
            and int(r['N']) == n and int(r['K']) == k
            and int(r['nhead']) == num_ops_per_hbm
            and int(r['dbyte']) == dbyte
            and bool_value(r['power_constraint']) == power_constraint
            and abs(float(r['sparsity']) - sparsity) < 1e-9
            and r['pim_type'] == pim_type.name
        ]
        if not row:
            return self.run(pim_type, layer, power_constraint)

        else:
            cycle = int(row[0]['cycle'])
            mac = int(row[0]['mac'])
            softmax = int(row[0]['softmax'])
            mvgb = int(row[0]['mvgb'])
            mvsb = int(row[0]['mvsb'])
            wrgb = int(row[0]['wrgb'])
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 1

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic
