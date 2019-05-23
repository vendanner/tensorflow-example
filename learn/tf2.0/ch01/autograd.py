import tensorflow as tf

x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    y = a**2 *x +b*x +c

def main():
    """
    对于函数 y = a**2 *x +b*x +c；求a,b,c 梯度
    dy_da = 2ax = 4
    dy_db = x = 1
    dy_dc = 1
    :return:
    """
    [dy_da, dy_db, dy_dc] = tape.gradient(y,[a,b,c])
    # tf.Tensor(4.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32)
    print(dy_da, dy_db, dy_dc)

if __name__ == "__main__":
    """
    """
    main()