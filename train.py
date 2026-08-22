import argparse
import math
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=None,
                    help='Decay step in examples; default is 20 training epochs')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='Decay rate for lr decay [default: 0.5, paper protocol]')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for reproducible training [default: 0]')
parser.add_argument('--jitter_sigma', type=float, default=0.02,
                    help='Gaussian jitter standard deviation [default: 0.02]')
parser.add_argument('--disable_augment', action='store_true',
                    help='Disable paper-style up-axis rotation and jitter')
parser.add_argument('--train_sampling', choices=['random', 'head'], default='random',
                    help='How to select NUM_POINT points from prepared HDF5 clouds '
                         '[default: random per training epoch]')
parser.add_argument('--legacy_fc1_dropout', action='store_true',
                    help='Enable the historical extra dropout after the 512-D layer')
parser.add_argument('--bn_decay_step_multiplier', type=float, default=2.0,
                    help='BN decay step as a multiple of the 20-epoch LR step')
parser.add_argument('--max_train_batches', type=int, default=None,
                    help='Optional per-file training batch limit for smoke tests')
parser.add_argument('--max_eval_batches', type=int, default=None,
                    help='Optional per-file evaluation batch limit for smoke tests')
parser.add_argument('--gpu_memory_fraction', type=float, default=0.20,
                    help='Maximum fraction of one GPU memory to reserve [default: 0.20]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MAX_TRAIN_BATCHES = FLAGS.max_train_batches
MAX_EVAL_BATCHES = FLAGS.max_eval_batches
GPU_MEMORY_FRACTION = FLAGS.gpu_memory_fraction
if not 0.0 < GPU_MEMORY_FRACTION <= 1.0:
    parser.error('--gpu_memory_fraction must be in (0, 1]')
if FLAGS.bn_decay_step_multiplier <= 0:
    parser.error('--bn_decay_step_multiplier must be positive')

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

MODEL = importlib.import_module(FLAGS.model) # import network module
if hasattr(MODEL, 'USE_FC1_DROPOUT'):
    MODEL.USE_FC1_DROPOUT = FLAGS.legacy_fc1_dropout
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def count_h5_samples(files):
    total = 0
    for filename in files:
        with h5py.File(filename, 'r') as handle:
            total += int(handle['label'].shape[0])
    return total


TRAIN_NUM_SAMPLES = count_h5_samples(TRAIN_FILES)
if DECAY_STEP is None:
    DECAY_STEP = TRAIN_NUM_SAMPLES * 20

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
# Match the public PointNet schedule: BN decay changes on a slower 40-epoch
# time scale while the learning rate halves every 20 epochs.
BN_DECAY_DECAY_STEP = float(DECAY_STEP * FLAGS.bn_decay_step_multiplier)
BN_DECAY_CLIP = 0.99

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=None)
            best_saver = tf.train.Saver(max_to_keep=1)
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_accuracy = -1.0
        best_class_accuracy = -1.0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_accuracy, eval_class_accuracy = eval_one_epoch(sess, ops, test_writer)
            
            # Keep a continuously updated checkpoint, periodic epoch snapshots,
            # and the best validation checkpoint.  The original script saved
            # epoch 0,10,...,240 for a 250-epoch run and never saved the final
            # state, which made the reported model ambiguous.
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            if (epoch + 1) % 10 == 0 or epoch == MAX_EPOCH - 1:
                epoch_path = saver.save(sess, os.path.join(LOG_DIR, "model_epoch_%03d.ckpt" % (epoch + 1)))
                log_string("Model saved in file: %s" % epoch_path)
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                best_class_accuracy = eval_class_accuracy
                best_path = best_saver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"))
                log_string("Best model saved in file: %s (accuracy=%f, class_accuracy=%f)" %
                           (best_path, best_accuracy, best_class_accuracy))



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = provider.sample_point_cloud(
            current_data, NUM_POINT, random=(FLAGS.train_sampling == 'random'))
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        if MAX_TRAIN_BATCHES is not None:
            num_batches = min(num_batches, MAX_TRAIN_BATCHES)
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            batch_data = current_data[start_idx:end_idx, :, :]
            if FLAGS.disable_augment:
                augmented_data = batch_data
            else:
                rotated_data = provider.rotate_point_cloud(batch_data)
                augmented_data = provider.jitter_point_cloud(
                    rotated_data, sigma=FLAGS.jitter_sigma)
            feed_dict = {ops['pointclouds_pl']: augmented_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        # Evaluation uses the stable prefix of the prepared cloud.  Training
        # performs fresh random subsampling; changing the test subset would
        # add an avoidable source of metric variance.
        current_data = provider.sample_point_cloud(current_data, NUM_POINT,
                                                   random=False)
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = (file_size + BATCH_SIZE - 1) // BATCH_SIZE
        if MAX_EVAL_BATCHES is not None:
            num_batches = min(num_batches, MAX_EVAL_BATCHES)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx+1) * BATCH_SIZE, file_size)
            cur_batch_size = end_idx - start_idx
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3), dtype=current_data.dtype)
            batch_labels = np.zeros((BATCH_SIZE,), dtype=current_label.dtype)
            batch_data[:cur_batch_size] = current_data[start_idx:end_idx]
            batch_labels[:cur_batch_size] = current_label[start_idx:end_idx]

            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_labels,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val[:cur_batch_size] == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += (loss_val*cur_batch_size)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    seen_class = np.asarray(total_seen_class, dtype=np.float64)
    correct_class = np.asarray(total_correct_class, dtype=np.float64)
    class_accuracy = np.divide(correct_class, seen_class,
                               out=np.zeros_like(correct_class), where=seen_class != 0)
    mean_class_accuracy = float(np.mean(class_accuracy))
    accuracy = float(total_correct / float(total_seen))
    log_string('eval avg class acc: %f' % mean_class_accuracy)
    return accuracy, mean_class_accuracy
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
