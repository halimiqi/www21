import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
import pdb



class Optimizer(object):
    def __init__(self, preds, labels, model, num_nodes,
                 global_step, new_learning_rate, new_learning_rate_gen,
                 if_drop_edge = True, **kwargs):
        allowed_kwargs = {'placeholders'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        placeholders = kwargs.get("placeholders")
        noised_indexes = placeholders["noised_mask"]
        clean_indexes = placeholders["clean_mask"]
        # en_preds_sub = preds
        # en_labels_sub = labels
        self.ratio_loss_fea = FLAGS.ratio_loss_fea
        self.opt_op = 0  # this is the minimize function
        self.cost = 0  # this is the loss
        self.accuracy = 0  # this is the accuracy
        self.G_comm_loss = 0
        self.G_comm_loss_KL = 0
        self.D_loss = 0
        self.num_nodes = num_nodes
        self.if_drop_edge = if_drop_edge
        self.last_reg = tf.Variable(0,name = "last_reg", dtype = tf.float32, trainable=False)
        self.generate_optimizer = tf.train.RMSPropOptimizer(learning_rate= new_learning_rate_gen)
        self.discriminate_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        generate_varlist = [var for var in tf.trainable_variables() if (
                    'generate' in var.name) or ('encoder' in var.name)]  
        discriminate_varlist = [var for var in tf.trainable_variables() if 'discriminate' in var.name]
        if if_drop_edge == True:
            self.G_comm_loss = self.loss_cross_entropy_logits_features(model, noised_indexes, clean_indexes, self.G_comm_loss)
            self.G_min_op = self.generate_optimizer.minimize(self.G_comm_loss, global_step=global_step,var_list=generate_varlist)
            self.D_loss = self.dis_cutmin_loss_clean_feature(model)
            self.D_min_op = self.discriminate_optimizer.minimize(self.D_loss, global_step = global_step,
                                                                 var_list = discriminate_varlist)




    def loss_cross_entropy_logits_features(self, model,noised_indexes, clean_indexes,  G_comm_loss):
        """
        the loss of reconstruction loss
        """
        noised_indexes_2d = tf.stack([noised_indexes //self.num_nodes,
                                      noised_indexes % self.num_nodes], axis = -1)
        clean_indexes_2d = tf.stack([clean_indexes // self.num_nodes,
                                     clean_indexes % self.num_nodes], axis =-1)
        adj_ori = model.adj_ori_dense - \
            tf.matrix_diag(tf.diag_part(model.adj_ori_dense))
        clean_mask = clean_indexes_2d
        real_pred = tf.gather_nd(model.x_tilde_output_ori,clean_mask)
        fake_pred = tf.gather_nd(model.x_tilde_output_ori, noised_indexes_2d)
        # pdb.set_trace()
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(real_pred), logits = real_pred)
        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(fake_pred), logits = fake_pred)
        mean_loss_fake = tf.reduce_mean(loss_fake)
        mean_loss_fake = tf.where(tf.is_nan(mean_loss_fake), 0., mean_loss_fake)
        G_comm_loss = tf.reduce_mean(loss_real) + mean_loss_fake
        return G_comm_loss


    def dis_cutmin_loss_clean_feature(self, model):
        """
        The cutmin loss
        """
        ######## construct the new A for cutmin loss  ############
        A_xx = model.adj_ori_dense * tf.matmul(model.feature_dense,
                                               tf.transpose(model.feature_dense))
        #########################################################
        A_pool = tf.matmul(
            tf.transpose(tf.matmul(A_xx, model.realD_tilde)), model.realD_tilde)
        num = tf.diag_part(A_pool)

        D = tf.reduce_sum(A_xx, axis=-1)
        D = tf.matrix_diag(D)
        D_pooled = tf.matmul(
            tf.transpose(tf.matmul(D, model.realD_tilde)), model.realD_tilde)
        den = tf.diag_part(D_pooled)
        D_mincut_loss = -(1 / FLAGS.n_clusters) * (num / den)
        D_mincut_loss = tf.reduce_sum(D_mincut_loss)
        ## the orthogonal part loss
        St_S = (FLAGS.n_clusters / self.num_nodes) * tf.matmul(tf.transpose(model.realD_tilde), model.realD_tilde)
        I_S = tf.eye(FLAGS.n_clusters)
        ortho_loss = tf.square(tf.norm(St_S - I_S))
        D_loss = D_mincut_loss + FLAGS.mincut_r * ortho_loss
        return D_loss
    pass

