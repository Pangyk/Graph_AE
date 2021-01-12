
def print_graph(batch, bin_batch, nodes):
    graph_list = []
    r_n_list = []
    t_n_list = []
    curr = 0
    sum_v = 0
    for i in nodes:
        if batch[i].item() != curr:
            if sum_v > 0:
                graph_list.append(curr)
                r_n_list.append(sum_v)
                t_n_list.append(bin_batch[curr].item())
            curr = batch[i].item()
            sum_v = 1
        else:
            sum_v += 1

    print("graphs: ", len(graph_list))
    print("has nodes: ", len(r_n_list))
    print("total nodes: ", len(t_n_list))

    with open("data/graph.txt", 'w') as f:
        f.write(str(graph_list))
        f.write("\n")
        f.write(str(r_n_list))
        f.write('\n')
        f.write(str(t_n_list))


def print_edge(original_e, bin_batch):
    s_list = []
    e_list = []
    o_s = original_e[0]
    o_e = original_e[1]
    for i in range(len(o_s)):
        s_n = o_s[i]
        if bin_batch[0] + bin_batch[1] <= s_n < bin_batch[0] + bin_batch[1] + bin_batch[2]:
            s_list.append(s_n.item())
            e_list.append(o_e[i].item())
        elif s_n >= bin_batch[0] + bin_batch[1] + bin_batch[2]:
            break
    with open("data/edge.txt", 'w') as f:
        f.write(str([s_list, e_list]))


def print_feature(recon, original_x):
    with open("data/feature.txt", 'w') as f:
        for i in range(10):
            f.write("r:" + str(recon[i].cpu().detach().numpy()))
            f.write("o:" + str(original_x[i].cpu().detach().numpy()))


def print_remain_nodes(sum_feature):
    n_list = []
    index = 0
    for n in sum_feature:
        if n > 0.0:
            n_list.append(index)
        index += 1
    with open("data/remain_nodes.txt", 'w') as f:
        f.write(str(n_list))
    return n_list
