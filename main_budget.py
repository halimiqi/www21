import tensorflow as tf

from utils import randomly_add_edges
from utils import get_noised_indexes, load_data_subgraphs
from utils import PSNR,WL_no_label
import datetime
import numpy as np
import scipy.sparse as sp
import time
import os
seed = 152   
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from mask_gvae import mask_gvae
from optimizer import Optimizer
from tqdm import tqdm

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
### BN  maybe cause the error

###
##### this is for gae part
flags.DEFINE_integer('n_clusters', 8, 'Number of epochs to train.')    # this one can be calculated according to labels
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1',64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
###########################
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('mincut_r', 0.01, 'The r parameters for the cutmin loss orth loss')   # ORTH LOSS
flags.DEFINE_string('model', 'mask_gvae', 'Model name.')
flags.DEFINE_string('dataset', 'PTC_MR', 'Dataset string.')
flags.DEFINE_float("noise_ratio" , 0.1, "the init of learn rate")
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
from tensorflow.python.client import device_lib
flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("latent_dim" , 16, "the dim of latent code")
flags.DEFINE_float("learn_rate_init" , 1e-03, "the init of learn rate")
flags.DEFINE_float("learn_rate_init_gen" , 1e-05, "the init of learn rate")
flags.DEFINE_integer("k", 20, "The edges to delete for the model")
flags.DEFINE_float('ratio_loss_fea', 1, 'the ratio of generate loss for features')
flags.DEFINE_boolean("train", True, "Training or Test")
###############################
if_train = FLAGS.train
cv_index = int(if_train)
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
gpu_id = 1
###################################
### read and process the graph
model_str = FLAGS.model
dataset_str = FLAGS.dataset
noise_ratio = 0.1
size = 0.5


n_class = FLAGS.n_clusters

def get_new_adj(feed_dict, sess, model, noised_index, adj_new, k, num_node):
    x_tilde = model.x_tilde.eval(session=sess, feed_dict=feed_dict)
    new_adj = adj_new.copy()
    noised_index = np.array(noised_index)
    if (len(noised_index) == 0) or (k == 0):
        return new_adj
    row = noised_index // adj_new.shape[0]
    col = noised_index % adj_new.shape[0]
    possible_edges = np.stack([row,col], axis = 0)
    noised_edges = []
    for i in range(possible_edges.shape[1]):
        if (possible_edges[0,i] > possible_edges[1,i]) and (possible_edges[0,i] < num_node) and (possible_edges[1,i] < num_node):
            noised_edges.append([possible_edges[:,i]])
    if len(noised_edges) == 0:
        return new_adj
    mask = np.ones_like(x_tilde)
    noised_edges = np.array(noised_edges)[:,0,:]
    mask[noised_edges[:,0], noised_edges[:,1]] = x_tilde[noised_edges[:,0], noised_edges[:,1]]
    mask_flat = mask.flatten()
    idxes_list = np.argsort(mask_flat)
    selected_idx = np.squeeze(idxes_list[:min(len(idxes_list), k)])
    row = selected_idx // x_tilde.shape[0]
    col = selected_idx % x_tilde.shape[0]
    new_adj[row, col] = 0
    new_adj[col, row] = 0
    new_adj = new_adj - sp.dia_matrix((new_adj.diagonal()[np.newaxis, :], [0]),
                                            shape=new_adj.shape)
    return new_adj

def add_noises_on_adjs(adj_list, num_nodes, noise_ratio = 0.1, ):
    noised_adj_list = []
    # add_idx_list = []
    adj_orig_list = []
    k_list = []
    for i in range(len(adj_list)):
        adj_orig = adj_list[i]
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                            shape=adj_orig.shape)  # delete self loop
        adj_orig.eliminate_zeros()
        # adj_new, add_idxes = add_edges_between_labels(adj_orig, int(noise_ratio* num_nodes[i]), y_train)
        adj_new,k_real = randomly_add_edges(adj_orig, int(noise_ratio* adj_orig[:num_nodes[i], :num_nodes[i]].sum() / 2), num_nodes[i])
        k_list.append(k_real)
        noised_adj_list.append(adj_new)
        # add_idx_list.append(add_idxes)
        adj_orig_list.append(adj_orig)
    return noised_adj_list, adj_orig_list, k_list

def get_new_feature(feed_dict, sess,flip_features_csr, feature_entry, model):
    new_indexes = model.flip_feature_indexes.eval(session = sess, feed_dict = feed_dict)
    flip_features_lil = flip_features_csr.tolil()
    for index in new_indexes:
        for j in feature_entry:
            flip_features_lil[index, j] = 1 - flip_features_lil[index, j]
    return flip_features_lil.tocsr()
# Train model
def train():
    ## add noise label
    train_adj_list, train_adj_orig_list, train_k_list = add_noises_on_adjs(train_structure_input, train_num_nodes_all)
    test_adj_list, test_adj_orig_list, test_k_list = add_noises_on_adjs(test_structure_input, test_num_nodes_all)

    adj = train_adj_list[0]
    features_csr = train_feature_input[0]
    features = sparse_to_tuple(features_csr.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_adj_orig_list[0]
    adj_label = train_adj_list[0] + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    num_nodes = adj.shape[0]

    adj_norm, adj_norm_sparse = preprocess_graph(adj)

    ############
    global_steps = tf.get_variable('global_step', trainable=False, initializer=0)
    new_learning_rate_dis = tf.train.exponential_decay(FLAGS.learn_rate_init, global_step=global_steps, decay_steps=100,
                                                   decay_rate=0.95)
    new_learning_rate_gen = tf.train.exponential_decay(FLAGS.learn_rate_init_gen, global_step=global_steps, decay_steps=100,
                                                       decay_rate=0.95)
    new_learn_rate_value = FLAGS.learn_rate_init
    ## set the placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32, name= "ph_features"),
        'adj': tf.sparse_placeholder(tf.float32,name= "ph_adj"),
        'adj_orig': tf.sparse_placeholder(tf.float32, name = "ph_orig"),
        'dropout': tf.placeholder_with_default(0.3, shape=(), name = "ph_dropout"),
        'clean_mask': tf.placeholder(tf.int32),
        'noised_mask': tf.placeholder(tf.int32),
        'noised_num':tf.placeholder(tf.int32),
        'node_mask':tf.placeholder(tf.float32)
    }
    # build models
    model = None
    adj_clean = adj_orig.tocoo()
    adj_clean_tensor = tf.SparseTensor(indices =np.stack([adj_clean.row,adj_clean.col], axis = -1),
                                       values = adj_clean.data, dense_shape = adj_clean.shape )
    if model_str == "mask_gvae":
        model = mask_gvae(placeholders, num_features, num_nodes, features_nonzero,
                       new_learning_rate_dis, new_learning_rate_gen,
                       adj_clean = adj_clean_tensor, k = int(adj.sum()*noise_ratio))
        model.build_model()
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    opt = 0
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'mask_gvae':
            opt = Optimizer(preds=tf.reshape(model.x_tilde, [-1]),
                                  labels=tf.reshape(
                                      tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
                                      [-1]),
                                  model=model,
                                  num_nodes=num_nodes,
                                  global_step=global_steps,
                                  new_learning_rate = new_learning_rate_dis,
                                  new_learning_rate_gen = new_learning_rate_gen,
                                  placeholders = placeholders
                                  )
    # init the sess
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ### initial clean and noised_mask
    clean_mask = np.array([1,2,3,4,5])
    noised_mask = np.array([6,7,8,9,10])
    noised_num = noised_mask.shape[0] / 2
    # ##################################
    feed_dict = construct_feed_dict(adj_norm, adj_label, features,clean_mask, noised_mask,noised_num,  placeholders)
    node_mask = np.ones([num_nodes, n_class])
    node_mask[train_num_nodes_all[0]:, :] = 0
    feed_dict.update({placeholders['node_mask']:node_mask})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # #####################################################
    if if_train:
        for epoch in range(FLAGS.epochs):
            for i in tqdm(range(len(train_feature_input))):
                train_one_graph(train_adj_list[i], train_adj_orig_list[i], train_feature_input[i], train_num_nodes_all[i], train_k_list[i], model, opt, placeholders,sess,new_learning_rate_gen,feed_dict, epoch, i)
        saver = tf.train.Saver()
        # saver.save(sess, "./checkpoints/{}/model.ckpt".format(cv_index))
        saver.save(sess, "./checkpoints/{}.ckpt".format(dataset_index))
        print("Optimization Finished!")
        psnr_list = []
        wls_list = []
        for i in range(len(test_feature_input)):
            psnr, wls = test_one_graph(test_adj_list[i], test_adj_orig_list[i],test_feature_input[i],test_num_nodes_all[i],test_k_list[i] , model, placeholders, sess, feed_dict)
            psnr_list.append(psnr)
            wls_list.append(wls)
    # new_adj = get_new_adj(feed_dict,sess, model)
    else:
      saver = tf.train.Saver()
      # saver.restore(sess, "./checkpoints/{}/model.ckpt".format(cv_index))
      saver.restore(sess, "./checkpoints/{}.ckpt".format(dataset_index))
      psnr_list = []
      wls_list = []
      for i in range(len(test_feature_input)):
          psnr, wls = test_one_graph(test_adj_list[i],test_adj_orig_list[i], test_feature_input[i], test_num_nodes_all[i], test_k_list[i], model, placeholders, sess, feed_dict)
          psnr_list.append(psnr)
          wls_list.append(wls)
    ##################################
    ################## the PSRN and WL #########################
    print("#"*15)
    print("The PSNR is:")
    psnr_list = [x for x in psnr_list if x != float("inf")] ## here isa bug, we can not check it
    print(np.mean(psnr_list))
    print("The WL is :")
    print(np.mean(wls_list))
    return psnr,wls

def train_one_graph(adj,adj_orig, features_csr ,num_node, k_num ,model, opt,placeholders, sess,new_learning_rate,feed_dict, epoch, graph_index):
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # delete self loop
    adj_orig.eliminate_zeros()
    adj_new  = adj
    row_sum = adj_new.sum(1).A1
    row_sum = sp.diags(row_sum)
    L = row_sum - adj_new
    ori_Lap = features_csr.transpose().dot(L).dot(features_csr)
    ori_Lap_trace = ori_Lap.diagonal().sum()
    ori_Lap_log = np.log(ori_Lap_trace)
    features = sparse_to_tuple(features_csr.tocoo())
    adj_norm, adj_norm_sparse = preprocess_graph(adj_new)
    adj_norm_sparse_csr = adj_norm_sparse.tocsr()
    adj_label = adj_new + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    ############
    ## set the placeholders
    # build models
    adj_clean = adj_orig.tocoo()
    adj_clean_tensor = tf.SparseTensor(indices =np.stack([adj_clean.row,adj_clean.col], axis = -1),
                                       values = adj_clean.data, dense_shape = adj_clean.shape )
    # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    ### initial clean and noised_mask
    clean_mask = np.array([1,2,3,4,5])
    noised_mask = np.array([6,7,8,9,10])
    noised_num = noised_mask.shape[0] / 2
    ##################################
    #
    feed_dict.update({placeholders["adj"]: adj_norm})
    feed_dict.update({placeholders["adj_orig"]: adj_label})
    feed_dict.update({placeholders["features"]: features})
    node_mask = np.ones([adj.shape[0], n_class])
    node_mask[num_node:, :] = 0
    feed_dict.update({placeholders['node_mask']: node_mask})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    model.k = k_num
    #####################################################
    t = time.time()
    ########
    # last_reg = current_reg
    if epoch > int(FLAGS.epochs / 2):  ## here we can contorl the manner of new model
        _= sess.run([opt.G_min_op], feed_dict=feed_dict,options=run_options)

    else:
        _, x_tilde = sess.run([opt.D_min_op, model.realD_tilde], feed_dict = feed_dict, options=run_options)
        if epoch == int(FLAGS.epochs / 2):
            noised_indexes, clean_indexes = get_noised_indexes(x_tilde, adj_new, num_node)
            feed_dict.update({placeholders["noised_mask"]: noised_indexes})
            feed_dict.update({placeholders["clean_mask"]: clean_indexes})
            feed_dict.update({placeholders["noised_num"]: len(noised_indexes)/2})
    ##
    if epoch % 1 == 0 and graph_index == 0:
        if epoch > int(FLAGS.epochs / 2):
            print("This is the generation part")
        else:
            print("This is the cluster mask part")
        print("Epoch:", '%04d' % (epoch + 1),
              "time=", "{:.5f}".format(time.time() - t))
        G_loss,D_loss, new_learn_rate_value = sess.run([opt.G_comm_loss,opt.D_loss,new_learning_rate],feed_dict=feed_dict,  options = run_options)
        print("Step: %d,G: loss=%.7f ,L_u: loss= %.7f, LR=%.7f" % (epoch, G_loss,D_loss + 1, new_learn_rate_value))
        ##########################################
    return

def test_one_graph(adj , adj_orig, features_csr, num_node, k_num ,model,placeholders, sess, feed_dict):
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)  # delete self loop
    adj_orig.eliminate_zeros()
    adj_new = adj
    row_sum = adj_new.sum(1).A1
    row_sum = sp.diags(row_sum)
    L = row_sum - adj_new
    ori_Lap = features_csr.transpose().dot(L).dot(features_csr)
    ori_Lap_trace = ori_Lap.diagonal().sum()
    ori_Lap_log = np.log(ori_Lap_trace)
    features = sparse_to_tuple(features_csr.tocoo())
    adj_label = adj_new + sp.eye(adj.shape[0])
    adj_label_sparse = adj_label
    adj_label = sparse_to_tuple(adj_label)
    adj_clean = adj_orig.tocsr()
    # pdb.set_trace()
    k_num = int(k_num*size/noise_ratio)
    if k_num !=0:
        # max_k = int((adj.sum()-adj_orig.sum())/2)
        # k_num = min(int(k_num*size/noise_ratio),max_k)
        adj_norm, adj_norm_sparse = preprocess_graph(adj_new)
        feed_dict.update({placeholders["adj"]: adj_norm})
        feed_dict.update({placeholders["adj_orig"]: adj_label})
        feed_dict.update({placeholders["features"]: features})
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        model.k = k_num
        # feed_dict = construct_feed_dict(adj_norm, adj_label, features, clean_mask, noised_mask, noised_num, placeholders)
        x_tilde = sess.run(model.realD_tilde, feed_dict=feed_dict, options=run_options)
        noised_indexes, clean_indexes = get_noised_indexes(x_tilde, adj_new, num_node)
        feed_dict.update({placeholders["noised_mask"]: noised_indexes})
        feed_dict.update({placeholders["clean_mask"]: clean_indexes})
        feed_dict.update({placeholders["noised_num"]: len(noised_indexes) / 2})
        test1 = model.test_new_indexes.eval(session=sess, feed_dict=feed_dict)
        test0 = model.test_noised_index.eval(session=sess, feed_dict=feed_dict)
        # print("########")
        # print(test0)
        # print(test1)
        # print(k_num)
        # print(len(noised_indexes))
        new_adj = get_new_adj(feed_dict, sess, model,noised_indexes, adj_new, k_num, num_node)
    else:
        # new_adj = adj_clean
        new_adj = adj
    new_adj_sparse = sp.csr_matrix(new_adj)

    psnr = PSNR(adj_clean[:num_node, :num_node], new_adj_sparse[:num_node, :num_node])
    wls = WL_no_label(adj_clean[:num_node, :num_node], new_adj_sparse[:num_node, :num_node])
    return psnr, wls



FLAGS = flags.FLAGS
if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    ## Load datasets
    # dataset_index = "IMDB-BINARY"
    # IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, MUTAG, PTC_MR   # here is the possible dataset
    dataset_index = 'PTC_MR'
    train_structure_input, train_feature_input, train_y, \
    train_num_nodes_all, test_structure_input, test_feature_input, \
    test_y, test_num_nodes_all = load_data_subgraphs(dataset_index, train_ratio=0.9)
    with open("results/results_%d_%s.txt"%(FLAGS.k, current_time), 'w+') as f_out:
        f_out.write( 'PSNR'+ ' ' + 'WL' + "\n")
        for i in range(1):
            psnr,wls = train()
            f_out.write(str(psnr)+ ' '+str(wls) + "\n")
    print(dataset_index)
    print(current_time)
