import numpy as np
import scipy
import optimizer
import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse,InnerProductDecoder, FullyConnect, Graphite, \
    GraphiteSparse,Scale,Dense,GraphiteSparse_simple, Graphite_simple,GraphConvolutionSparse_denseadj, GraphConvolution_denseadj
from ops import batch_normal
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from optimizer import Optimizer
flags = tf.app.flags
FLAGS = flags.FLAGS


class mask_gvae(object):
    def __init__(self,placeholders, num_features,num_nodes, features_nonzero,
                 learning_rate_init ,if_drop_edge = True, **kwargs):
        allowed_kwargs = {'name', 'logging', 'indexes_add', 'adj_clean', 'k'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.input_dim = num_features
        self.inputs = placeholders['features']
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adj_ori = placeholders['adj_orig']
        self.features_nonzero = features_nonzero
        self.batch_size = FLAGS.batch_size
        self.latent_dim = FLAGS.latent_dim
        self.n_samples = num_nodes  # this is the number of nodes in the nodes
        self.n_clusters = FLAGS.n_clusters
        self.zp = tf.random_normal(shape=[self.n_samples, self.latent_dim])
        self.learning_rate_init = learning_rate_init
        self.if_drop_edge = if_drop_edge
        #######################################
        self.test_mask_adj = tf.one_hot(5, self.n_samples * self.n_samples,
                                        on_value = True,
                                        off_value = False,dtype = tf.bool)
        self.test_mask_feature = tf.one_hot(10, self.n_samples * self.input_dim,
                                            on_value = True,
                                            off_value = False,dtype= tf.bool)
        self.indexes_add_orig = placeholders['noised_mask']
        self.adj_clean = kwargs["adj_clean"]
        self.noised_num = placeholders["noised_num"]
        self.clean_indexes = placeholders["clean_mask"]
        self.placeholders = placeholders
        self.k = kwargs["k"]
        #######################################
        return

    def build_model(self):
        '''
        build the model of
        '''
        #### this is the first thing we have
        self.adj_dense = tf.sparse_tensor_to_dense(self.adj, default_value=0, validate_indices=False, name=None) # normalize adj
        self.realD_tilde = self.discriminate_mock_detect(self.inputs, self.adj_dense,
                                                             reuse=False)
        self.realD_tilde = self.realD_tilde * self.placeholders['node_mask']
        #######################
        # build the model
        self.z_x = self.encoder(self.inputs)  # incoder
        self.x_tilde = 0
        self.new_adj_outlist = []
        self.new_features_list = []
        self.reward_percent_list = []
        self.percentage_list_all = []
        self.percentage_fea = []
        self.new_adj_output = self.adj_dense
        self.adj_ori_dense = tf.sparse_tensor_to_dense(self.adj_ori, default_value=0, validate_indices=False, name=None) #A + I
        self.adj_clean_dense = tf.sparse_tensor_to_dense(self.adj_clean, default_value = 0, validate_indices = False)  ## no noise one
        self.x_tilde = self.generate_dense(self.z_x, self.z_x.shape[1], self.input_dim)
        self.x_tilde_output_ori = self.x_tilde
        if self.if_drop_edge != False:
            #######
            ######### prepare the graph for delete k edges
            ones = tf.ones_like(self.x_tilde, dtype=tf.float32)
            self.zeros = tf.zeros_like(self.x_tilde, dtype = tf.bool)
            self.feature_dense = tf.sparse_tensor_to_dense(self.inputs)
            self.ones_feature = tf.ones_like(self.feature_dense)
            max_value = tf.reduce_max(self.x_tilde)
            lower_bool_label = tf.linalg.band_part(self.adj_ori_dense, -1, 0)
            upper_ori_label = self.adj_ori_dense - lower_bool_label  # there is no diagnal
            upper_bool_label = tf.cast(upper_ori_label, tf.bool)
            new_adj_for_del = tf.where(upper_bool_label, x=self.x_tilde, y=ones * max_value, name="delete_mask")
            self.new_adj_for_del_test = max_value - new_adj_for_del
            new_adj_for_del = max_value - new_adj_for_del    # by this we put the no edge value to 0 and put the minimum value to the larges
            ori_adj_diag = tf.matrix_diag(tf.matrix_diag(self.adj_ori_dense))
            new_adj_diag = tf.matrix_diag(tf.matrix_diag_part(self.x_tilde))  # diagnal matrix
            ori_adj_diag = tf.reshape(ori_adj_diag, [-1])
            new_adj_flat = tf.reshape(self.x_tilde, [-1])
            ori_adj_flat = tf.reshape(self.adj_ori_dense, [-1])
            ### doing the softmax function
            new_adj_for_del_exp = tf.exp(new_adj_for_del)
            new_adj_for_del_exp = tf.where(upper_bool_label, x=new_adj_for_del_exp,
                                           y=tf.zeros_like(new_adj_for_del_exp), name="softmax_mask")
            new_adj_for_del_softmax = new_adj_for_del_exp / tf.reduce_sum(new_adj_for_del_exp)
            new_adj_for_del_softmax = tf.reshape(new_adj_for_del_softmax, [-1])
            ############ delete k edges
            self.new_adj = self.delete_edge(self.x_tilde,
                                                 self.indexes_add_orig,self.noised_num, self.k)
            # self.new_feature_prob = self.generate_feature_prob(self.z_x, self.feature_dense, self.input_dim, self.input_dim * 2)
            # self.flip_feature_indexes = self.flip_features(self.clean_indexes, k = FLAGS.k_features)
        return


    def delete_edge(self, x_tilde, noised_index,noised_num, k):
        """
        delete_mask_idx_sparse
        select the minimum k edges to delete
        """

        row = noised_index // self.n_samples
        col = noised_index % self.n_samples
        selected = tf.where(tf.greater(col, row))
        noised_index = tf.gather(noised_index, selected)
        new_row = noised_index // self.n_samples
        new_col = noised_index % self.n_samples
        noised_index = (tf.stack([new_row,new_col], axis = -1))[:,0,:]
        #noised_index = tf.squeeze(tf.stack([new_row,new_col], axis = -1))
        sampled_dist = tf.gather_nd(x_tilde, noised_index)
        sampled_dist = tf.reshape(sampled_dist, [-1])
        sampled_dist = tf.reduce_max(sampled_dist) - sampled_dist   # delete the edges with smallest edges
        sampled_dist = tf.nn.softmax(sampled_dist)
        _, indexes = tf.nn.top_k(sampled_dist, tf.minimum(noised_num, k))
        new_indexes = tf.gather(noised_index, indexes, axis = 0)
        self.test_noised_index = noised_index
        self.test_new_indexes = new_indexes
        self.test_sampled_dist = sampled_dist
        row_idx = new_indexes[:,0]
        col_idx = new_indexes[:,1]
        indices = tf.stack([row_idx, col_idx], axis = -1)
        indices_other = tf.stack([col_idx, row_idx], axis = -1)
        indices = tf.concat([indices, indices_other], 0)
        indices = tf.cast(indices, tf.int64)
        shape = [self.n_samples, self.n_samples]
        delete_mask_idx_sparse = tf.SparseTensor(indices,tf.ones_like(indices[:,0], dtype = tf.int64), shape)
        delete_mask_idx_sparse = tf.cast(delete_mask_idx_sparse, tf.bool)
        new_adj = tf.where(tf.sparse.to_dense(delete_mask_idx_sparse, default_value = False, validate_indices = False),
                                              x = tf.cast(self.zeros, tf.float32), y = self.adj_ori_dense)
        return new_adj


    # def flip_features(self,clean_indexes, k = 10,  reuse = tf.AUTO_REUSE):
    #     with tf.variable_scope("generate_flip_fea") as scope:
    #         feature_dense = self.feature_dense[:,:FLAGS.k_features_dim]
    #         node_feature = feature_dense - self.new_feature_prob
    #         node_feature = tf.norm(node_feature, axis = -1)
    #         _, indexes = tf.nn.top_k(node_feature, tf.minimum(self.n_samples, k))
    #     return indexes

    def encoder(self, inputs):
        with tf.variable_scope('encoder') as scope:
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging, name = "encoder_conv1")(inputs)

            self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.latent_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,name = "encoder_conv2")(self.hidden1)

            self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.latent_dim,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging,name = "encoder_conv3")(self.hidden1)

            z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.latent_dim]) * tf.exp(
                self.z_log_std)  # middle hidden layer
        return z


    def generate_dense(self, input_z, input_dim, graph_dim, reuse = False):
        input_dim = int(input_dim)
        with tf.variable_scope('generate') as scope:
            if reuse == True:
                scope.reuse_variables()
            update_temp = []
            ## the element wise product to replace the current inner product with size n^2*d
            for i in range(0, self.n_samples):
                update_temp.append(input_z[i, :] * input_z)
            final_update = tf.stack(update_temp, axis=0)
            reconstructions = tf.layers.dense(final_update, 1,use_bias=False, activation = tf.nn.sigmoid, name="gen_dense2")
            reconstructions = tf.squeeze(reconstructions)
        return reconstructions

    # def generate_feature_prob(self, input_z,input_feature,input_dim, hidden1_dim, reuse = False):
    #     with tf.variable_scope('generate_feature') as scope:
    #         if reuse == True:
    #             scope.reuse_variables()
    #         h1 = FullyConnect(output_size=hidden1_dim, scope="generate_feature_full1")(input_z)
    #         h1 = tf.nn.relu(h1)
    #         H = FullyConnect(output_size=FLAGS.k_features_dim, scope="generate_feature_full2")(h1)
    #         N = tf.nn.softmax(H, axis = -1)  # then shape of N is n*d
    #         input_feature = input_feature[:, :FLAGS.k_features_dim]
    #         ones = tf.ones_like(input_feature)
    #         X =input_feature * (1 - N) + (ones - input_feature) * N
    #     return X
    def discriminate_mock_detect(self, inputs,new_adj,  reuse = False):
        with tf.variable_scope('discriminate') as scope:
            if reuse == True:
                scope.reuse_variables()

            self.dis_hidden = GraphConvolutionSparse_denseadj(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=new_adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging, name ="dis_conv1_sparse")((inputs, new_adj))


            self.dis_z_mean = GraphConvolution_denseadj(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=new_adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging, name='dis_conv2')((self.dis_hidden, new_adj))
            ############################
            self.dis_z_mean_norm = tf.nn.softmax(self.dis_z_mean, axis = -1)
            ############################
            self.dis_fully1 =tf.nn.relu(batch_normal(FullyConnect(output_size=256, scope='dis_fully1')(self.dis_z_mean),scope='dis_bn1', reuse = reuse))
            self.dis_output = FullyConnect(output_size = self.n_clusters, scope='dis_fully2')(self.dis_fully1)
            # the softmax layer for the model
            self.dis_output_softmax = tf.nn.softmax(self.dis_output, axis=-1)
        return self.dis_output_softmax



    def get_edge_indexes(self, x_tilde, adj_dense, n_clusters):
        node_comm = tf.argmax(x_tilde, axis = -1)
        in_clus_idx_list = []
        adj_diag = adj_dense - tf.matrix_diag(tf.diag_part(adj_dense))
        for i in range(n_clusters):
            selected_node_idx = tf.where(tf.equal(node_comm,
                                                  tf.constant(i, dtype = tf.int64)))
            indices = tf.expand_dims(selected_node_idx, 1)
            values = tf.ones_like(selected_node_idx)
            shape = selected_node_idx.shape
            import pdb; pdb.set_trace()
            sparse_onehot_tensor = tf.SparseTensor(indices, values, shape)
            onehot_dense = tf.sparse_tensor_to_dense(sparse_onehot_tensor)
            mask = tf.matmul(onehot_dense.transpose(), onehot_dense)
            in_clus_idx_list.append(selected_node_idx)
        return

    def intersect_edges(self,indexes_delete, indexes_orig):
        intersect = tf.sets.set_intersection(indexes_delete[None,:],
                                             indexes_orig[None,:])
        intersect = tf.sparse_tensor_to_dense(intersect)
        length = tf.ones_like(intersect)
        return tf.reduce_sum(length)
    pass
