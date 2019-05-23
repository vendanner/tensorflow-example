import tensorflow as tf
import timeit


with tf.device("/gpu:0"):
    a_gpu = tf.random.normal([10000,1000])
    b_gpu = tf.random.normal([1000,2000])
    print(a_gpu.device,b_gpu.device)

def gpu_run():
    """

    :return:
    """
    with tf.device("/gpu:0"):
        c_gpu = tf.matmul(a_gpu,b_gpu)
    return c_gpu

# with tf.device("/cpu:0"):
#     """
#     """
#     a = tf.random.normal([10000,1000])
#     b = tf.random.normal([1000,2000])
#     print(a.device,b.device)

def cpu_run():
    """
    计算 10000 * 1000 向量 和 1000 * 2000 向量 点积耗时
    :return:
    """
    with tf.device("/cpu:0"):
        a = tf.random.normal([10000, 1000])
        b = tf.random.normal([1000, 2000])
        c = tf.matmul(a,b)
    return c

def main():
    """

    :return:
    """
    print("first: ",timeit.timeit(cpu_run,number=10))
    print("second: ", timeit.timeit(cpu_run, number=10))


if __name__ == "__main__":
    """
    """
    main()