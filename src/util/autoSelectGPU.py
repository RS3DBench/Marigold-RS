import subprocess


def get_gpu_info():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, encoding='utf-8')

    gpu_info = []
    for line in result.stdout.strip().split('\n'):
        index, mem_used, mem_total, gpu_util = map(int, line.split(', '))
        gpu_info.append({
            'index': index,
            'mem_used': mem_used,
            'mem_total': mem_total,
            'gpu_util': gpu_util,
            'mem_free': mem_total - mem_used
        })
    return gpu_info


def select_best_gpu():
    gpu_info = get_gpu_info()
    # gpu_info = [gpu for gpu in gpu_info if gpu['index'] == 2]
    gpu_info.sort(key=lambda x: (-x['mem_free'], x['gpu_util']))
    print("GPU info:")
    for gpu in gpu_info:
        print(gpu)
    best_gpu = gpu_info[0]['index']
    if best_gpu == 0 and len(gpu_info) > 1 and gpu_info[1]['mem_free'] == gpu_info[0]['mem_free']:
        print(
            f"GPU {gpu_info[0]['index']} and GPU {gpu_info[1]['index']} have the same free memory, choose GPU {gpu_info[1]['index']}.")
        best_gpu = gpu_info[1]['index']
    print(f"Choose GPU {best_gpu}")

    return best_gpu
    # print(f"Choose GPU 3")
    # return 3
