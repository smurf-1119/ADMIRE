import argparse

from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf, GpuType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("--worker_num", type=int, default=1)
    args = parser.parse_args()
    print("=" * 10 + f" Runing script {args.script} " + "=" * 10)

    # 资源设置
    # gpu_num = args.devices
    # cpu = gpu_num * 2
    # num = 2
    # memory = 1024 * 100 * gpu_num

    # master = ExecConf(
    #     cpu=cpu,
    #     memory=memory,
    #     gpu_num=gpu_num,
    #     num=num,
    #     gpu_type=GpuType.A100,
    # )

    worker_num = args.worker_num # 可以从0到n
    # all gpu_num = master_gpu_num * master_num + worker_gpu_num * worker_num
    master = ExecConf(cpu=10 * 4, memory=1024 * 500 * 8, gpu_num=8, num=1, gpu_type=GpuType.A100)
    worker = ExecConf(cpu=10 * worker_num * 4, memory=1024 * 500 * 8 * worker_num, gpu_num=8, num=worker_num, gpu_type=GpuType.A100)

    km_conf = KMConf(
        # 镜像，请根据自己需要和GPU环境选择
        # image="reg.docker.alibaba-inc.com/aii/aistudio:400318-20230717171959"
        image="reg.docker.alibaba-inc.com/aii/aistudio:aistudio-160895473-1249154481-1731856945918"
        )

    job = PythonJobBuilder(
        # 打包目录下所有文件
        source_root="./",
        # 容器内运行命令
        command=f"sh {args.script}",
        main_file="",
        master=master,
        worker=worker,
        # 任务提交到的应用
        # k8s_app_name="agiapps",
        k8s_app_name="amed",
        # k8s_app_name="apmktalgogpu",
        k8s_priority="high",
        km_conf=km_conf,
        # 使用pytorch
        runtime="pytorch",
        tag="type=SFT,basemodel=InternVL2"
    )

    job.run(enable_wait=False)


if __name__ == "__main__":
    main()