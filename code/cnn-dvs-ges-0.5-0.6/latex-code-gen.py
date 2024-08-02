import os
import subprocess

seed_list = [1111, 2222, 3333, 4444, 5555]
snn_ratio = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375]

log_folder_path = "./log/"

dataset = "dvsGesture"

rnn_type = "rnn"

def extract_one_performance(model, seed, ratio=None):
    if model == "rnn" or model == "lstm" or model == "snn":
        file_name = "train" + "_" + dataset + "_" + model + "_" + str(seed) + ".log"
    else:
        assert model == "ffs" or model == "hybrid"
        file_name = "train" + "_" + dataset + "_" + model + "_" + str(seed) + "_" + str(ratio) + ".log"
    cmd = "tail -n 3 " + log_folder_path + file_name
    print(cmd)
    result = subprocess.getoutput(cmd)
    perf = result.split("\n")[1].split(" ")[-1]
    #if results.find("test acc") > 0 or results.find("test ppl") > 0:
    #print(perf)
    return perf

#    print(result)


#for i in range(1, 6):
#    extract_one_performance("rnn", 1111 * i)

csv_str_ffs = "type, 1111, 2222, 3333, 4444, 5555, avg, squ\n"

csv_str_hybrid = "type, 1111, 2222, 3333, 4444, 5555, avg, squ\n"

if rnn_type == "rnn":
    rnn_str = "RNN"
else:
    rnn_str = "LSTM"

def compute_avg_squ(type, perf_list):
    
    tmp_str = ""
    tmp_str += type + ", "

    for p in perf_list:
        tmp_str += str(round(float(p), 2)) + ", "

    sum = 0.0
    for p in perf_list:
        sum += float(p)
    avg = sum / len(perf_list)
    
    squ = 0.
    for p in perf_list:
        squ += (float(p) - avg) ** 2
    squ = squ / len(perf_list)

    avg = round(avg, 2)
    squ = round(squ, 2)

    tmp_str += str(avg) + ", "
    tmp_str += str(squ) + ""

    tmp_str += "\n"
    return tmp_str


def gen_rnn_snn_line(model):
    global csv_str_hybrid, csv_str_ffs
    perf = []
    for i in seed_list:
        perf.append(extract_one_performance(model, i, ratio=None))
    latex_str = model
    for i in perf:
        latex_str += " & " + "\multicolumn{2}{c|}{" + i +"} " 

    # compute avg and stderr
    csv_line = compute_avg_squ(str(model), perf)

    csv_str_ffs += csv_line
    csv_str_hybrid += csv_line

    latex_str += " \\\ \hline \n"
    #print(latex_str)
    return latex_str

def gen_ratio_line(ratio):
    global csv_str_hybrid, csv_str_ffs
    ffs_perf = []
    hybrid_perf = []

    for i in seed_list:
        ffs_perf.append(extract_one_performance("ffs", i, ratio=ratio))
        hybrid_perf.append(extract_one_performance("hybrid", i, ratio=ratio))

    latex_str = str(ratio)

    if ratio == 0.5:
        latex_str += "0"

    ffs_csv_line = compute_avg_squ(latex_str, ffs_perf)
    hybrid_csv_line = compute_avg_squ(latex_str, hybrid_perf)

    csv_str_ffs += ffs_csv_line
    csv_str_hybrid += hybrid_csv_line

    for i in range(5):
        latex_str += " & " + ffs_perf[i] + " & " + "\\textbf{" + hybrid_perf[i]+ "} "
    latex_str += " \\\ \hline \n"
    return latex_str


prefix = "\
\\begin{table}[] \n\
\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|} \n\
\hline \n\
     & \multicolumn{2}{c|}{1111}  & \multicolumn{2}{c|}{2222}  & \multicolumn{2}{c|}{3333}  & \multicolumn{2}{c|}{4444}  & \multicolumn{2}{c|}{5555}  \\\ \hline \n"

prefix += gen_rnn_snn_line(rnn_type)

for i in snn_ratio:
    prefix += gen_ratio_line(i)

prefix += gen_rnn_snn_line("snn")

prefix += "\end{tabular}\n\end{table}\n"

print(prefix)

print("\n\n\n")

os.system("mkdir -p results")

with open("./results/" + dataset + "-" + rnn_str + "-ffs.csv", 'w') as fp:
    print(csv_str_ffs, file=fp)

with open("./results/" + dataset + "-" + rnn_str + "-hybrid.csv", 'w') as fp:
    print(csv_str_hybrid, file=fp)
