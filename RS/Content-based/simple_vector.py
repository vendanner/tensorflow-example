"""
基于 moive 的特征做推荐
1、根据 user_moives 和 moive_features，得到每个用户对于每个特征的总和以及所有特征的总和
2、即可得到每个用户各自每个特征的权重(每个特征的总和/所有特征的总和)
3、利用每个用户各自每个特征的权重 * 每部电影的特征值 = 每个用户对每个电影的打分(模型预测的 user_moives)
4、去除模型预测出的 user_moives 中已打分的电影，然后对每个用户取出top k 电影
5、top k 电影就是要推荐给用户电影
"""
import tensorflow as tf
import numpy as np

print(tf.__version__)

def find_user_top_feats(users_feats,user_index,num_feats,features):
    """
    根据用户的特征权重大小，返回特征字符串(权重从高到低排序)
    :param users_feats:
    :param user_index:
    :return:
    """
    # TopKV2(values=array([0.87435634, 0.77118243, 0.69864978, 0.47136589, 0.3439558 ]), indices=array([8, 2, 5, 9, 4]))
    # [1] 返回 indices
    feats_ind = tf.nn.top_k(users_feats[user_index],num_feats)[1]
    return tf.gather_nd(features,tf.expand_dims(feats_ind,axis =1))

def find_user_movies(ind,num_recommend,all_user_ratings_new,moives):
    """
    在 all_user_ratings_new 取 top num_recommend 后，返回电影名
    :param ind:
    :param num_recommend:
    :param all_user_ratings_new:
    :return:
    """
    moive_ind = tf.nn.top_k(all_user_ratings_new[ind],num_recommend)[1]
    return tf.gather_nd(moives,tf.expand_dims(moive_ind,axis =1))

user = ['Ryan','Danielle','Vijay','Chris']
moives = ['Star Wars','The Dark Knight','Shrek','The Incredibles','Blue','Memento']
features = ['Fantasy' ,'Action' ,'Cartoon' ,'Drama' ,'Comedy']
num_users = len(user)
num_moives = len(moives)
num_features = len(features)

# 4位用户对6部电影的评分，0 表示未看
user_moives  = tf.constant([[4,6,8,0,0,0],
                           [0,0,10,0,8,3],
                           [0,6,0,0,3,7],
                           [10,9,0,5,0,2]])
# 6部电影各自的五个特征值：`Fantasy` 、`Action` 、`Cartoon` 、`Drama` 、`Comedy`
moive_features = tf.constant([[1,1,0,0,1],
                              [1,1,0,0,0],
                              [0,0,1,1,0],
                              [1,0,1,1,0],
                              [0,0,0,0,1],
                              [1,0,0,0,1]])

user_moives = tf.cast(user_moives,dtype=tf.float32)
moive_features =tf.cast(moive_features,dtype = tf.float32)

# tf.transpose(user_moives)[:,i] 每个人对6部电影的评分
# wgdt_feature_matrics 每个人对每部电影的每个特征评分
wgdt_feature_matrics = [tf.expand_dims(tf.transpose(user_moives)[:,i],axis =1) * moive_features for i in range(num_users)]
# 组装 wgdt_feature_matrics 矩阵数组为整个矩阵
user_moives_feats = tf.stack(wgdt_feature_matrics,axis =0)

# 加和:每个用户对某个电影特征评分总和
user_moives_feats_sum = tf.reduce_sum(user_moives_feats,axis=1)
# 加和：每个用户对所有电影特征评分总和
user_moives_feats_total = tf.reduce_sum(user_moives_feats_sum,axis=1)

# 得到每个用户对于特征的权重系数
users_feats = tf.stack([ user_moives_feats_sum[i,:]/ user_moives_feats_total[i]for i in range(num_users)],axis =0)
# 用户特征权重字符串从高到低排序
users_topfeats ={}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_users):
        top_feats = sess.run(find_user_top_feats(users_feats,i,num_features,features))
        users_topfeats[user[i]] = list(top_feats)

print(users_topfeats)

# 利用每个用户各自每个特征的权重 * 每部电影的特征值 = 每个用户对每个电影的打分(模型预测的 user_moives)
user_ratings = [tf.map_fn(lambda u: tf.tensordot(users_feats[i],u,axes = 1),moive_features) for i in range(num_users)]
all_user_ratings = tf.stack(user_ratings)
# np.inf 无穷大；-np.inf 负无穷大
# user_moives 中原本有值直接赋值负无穷大，无值取 all_user_ratings 数值,为了后续直接取 top k
all_user_ratings_new = tf.where(tf.equal(user_moives,tf.zeros_like(user_moives)),all_user_ratings,-np.inf * tf.ones_like(tf.cast(user_moives,tf.float32)))

user_topmovies = {}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 每个用户需要推荐的电影数 = 用户未评分的电影数
    num_recommend = tf.reduce_sum(tf.cast(tf.equal(user_moives,tf.zeros_like(user_moives)),dtype = tf.float32),axis =1)
    for ind in range(num_users):
        top_movies = sess.run(find_user_movies(ind,tf.cast(num_recommend[ind],dtype = tf.int32),all_user_ratings_new,moives))
        user_topmovies[user[ind]] = list(top_movies)

print(user_topmovies)