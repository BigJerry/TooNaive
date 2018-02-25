import time
import tensorflow as tf
import numpy as np
from PIL import Image
import gym
import random

def timing(func):
    def tictoc(*args,**kwargs):
        start = time.time()
        ret = func(*args,**kwargs)
        record = time.time() - start
        print("Function '%s' consumed %f s"%(func.__name__,record))
        return ret
    
    return tictoc

class StateSequence(object):
    def __init__(self,env):
        self.env = env
        self.list_actions = []
        self.list_states = []
        self.list_tflag = []
        
    @property
    def len_state(self):
        return len(self.list_states)
        
    def reset(self):
        init_state = self.env.reset()
        self.list_actions = []
        self.list_states = []
        self.list_tflag = []
        self.list_states.append(init_state)
    
    def expand(self,act,next_state,t_flag):
        next_seq = StateSequence(self.env)
        
        next_action_list = self.list_actions.copy()
        if len(next_action_list) > 3:
            next_action_list.pop(0)
        next_action_list.append(act)
        
        next_state_list = self.list_states.copy()
        if len(next_state_list) > 3:
            next_state_list.pop(0)
        next_state_list.append(next_state)
        
        next_tflag_list = self.list_tflag.copy()
        if len(next_tflag_list) > 3:
            next_tflag_list.pop(0)
        next_tflag_list.append(t_flag)        
        
        next_seq.list_actions = next_action_list
        next_seq.list_states = next_state_list
        next_seq.list_tflag = next_tflag_list
        return next_seq
    
    def update(self,act,next_state,t_flag):
        def pop_if_overflow(ls):
            if len(ls) > 3:
                ls.pop(0)
        list(map(pop_if_overflow,[self.list_actions,self.list_states,self.list_tflag]))
        self.list_actions.append(act)
        self.list_states.append(next_state)
        self.list_tflag.append(t_flag)
    
    def copy(self):
        new_seq = StateSequence(self.env)
        new_seq.list_actions = self.list_actions.copy()
        new_seq.list_states = self.list_states.copy()
        new_seq.list_tflag = self.list_tflag.copy()
        return new_seq

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.s_ = []
        self.a = []
        self.r = []
        self.next_s_ = []
        
    def _save(self,lists,element):
        if len(lists)<self.capacity:
            lists.append(element)
        else:
            lists.pop(0)
            lists.append(element)
            
    def _get_sample(self,lists,idxs):
        ret = []
        for idx in idxs:
            ret.append(lists[idx])
        return ret
    
    def _make_sample_ret(self,ls):
        return tuple(zip(ls[0][:],ls[1][:],ls[2][:],ls[3][:]))
    
    @property
    def current_capacity(self):
        return len(self.a)
    
    def store(self,transition_tuple):
        """transition_tuple should have the format of 
        (processed sequence,action,reward,next processed sequence).
        only transition tuple that is of s_ list of length larger than 3 will 
        be stored."""
        if transition_tuple[0].len_state > 3:
            list(map(self._save,[self.s_,self.a,self.r,self.next_s_],transition_tuple))
            return True
        return False
    
    def sample(self,batch_size):                                                    
        """this method returns a tuple containing tuples of sampled transitions."""
        hitted_idxs = random.sample(range(self.current_capacity),batch_size)
        ret_ls = list(map(self._get_sample,[self.s_,self.a,self.r,self.next_s_],[hitted_idxs]*4))
        ret = self._make_sample_ret(ret_ls)
        return ret
    
class DQN(object):
    def __init__(self,sess,gamma,lr,batch_size):
        self.sess = sess
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self._init_graph()
        
    def _init_graph(self):
        self._construct()
        self.sess.run(tf.global_variables_initializer())
        
    def _flatten(self,layers):
        layers = tf.contrib.layers.flatten(layers)
        return layers
    
    def _weights(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
    
    def _bias(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
        
    def _construct(self):
        """Define what the graph of DQN is.This method was called once instantiate
        a DQN object"""
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,84,84,4])               
        self.label_info = tf.placeholder(tf.int32,None)
        self.label = tf.placeholder(tf.float32,None)
        with tf.name_scope("conv_1"):
            w_h1 = self._weights([8,8,4,16])
            b_h1 = self._bias([16])
            h1_conv = tf.nn.conv2d(self.x,w_h1,[1,4,4,1],'SAME') + b_h1
            h1 = tf.nn.relu(h1_conv)
        with tf.name_scope("conv_2"):
            w_h2 = self._weights([4,4,16,32])
            b_h2 = self._bias([32])
            h2_conv =tf.nn.conv2d(h1,w_h2,[1,2,2,1],'SAME') + b_h2
            h2 = tf.nn.relu(h2_conv)
        with tf.name_scope("fc"):
            h2_flat = self._flatten(h2)
            w_fc = self._weights([h2_flat.shape[1:].num_elements(),256])
            b_fc = self._bias([256])
            fc = tf.nn.relu(tf.matmul(h2_flat,w_fc) + b_fc)
        with tf.name_scope("output"):
            w_out = self._weights([256,4])
            b_out = self._bias([4])
            self.output = tf.matmul(fc,w_out) + b_out                          #self.output has shape of (1,4)
        with tf.name_scope("filter_output"):
            self.filtered_output = tf.gather_nd(self.output,indices=self.label_info)
        with tf.name_scope("loss"):
            loss = tf.losses.mean_squared_error(labels=self.label,
                                                predictions=self.filtered_output)
        with tf.name_scope("optimizer"):
            self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
                    
    def _compute_label(self,batch_transition):
        ret = []
        for piece in batch_transition:
            if piece[3].list_tflag[-1]:
                ret.append([piece[2],piece[1]])
            else:
                x = np.array(piece[3].list_states[:]).reshape((1,84,84,4))
                feed_dict = {self.x:x}
                max_act_Q_value = np.max(self.sess.run(self.output,feed_dict=feed_dict))
                y = piece[2] + self.gamma * max_act_Q_value
                ret.append([y,piece[1]])
        return np.array(ret).reshape((len(ret),2))
    
    def _make_input_batch_optimize(self,batch_transition):
        ret = []
        for piece in batch_transition:
            ret.append(piece[0].list_states[:])
        return np.array(ret).reshape((self.batch_size,84,84,4))
    
    def _make_input_batch_select_action(self,seq):
        return np.array(seq.list_states[-4:]).reshape((1,84,84,4))
    
    def select_action(self,seq):                                               #ToDo:implement other branch of selection strategy,which is random selection
        if len(seq.list_states) > 3:
            inputs = self._make_input_batch_select_action(seq)
            feed_dict = {self.x:inputs}
            return np.argmax(self.sess.run(self.output,feed_dict=feed_dict))   
        else:
            return random.randrange(0,4)

    def optimize(self,batch_transition):                                    
        """update network's parameters"""
        x = self._make_input_batch_optimize(batch_transition)
        info = self._compute_label(batch_transition)
        label = info[:,0].reshape((info[:,0].size,1))
        info1 = np.array(range(32)).reshape((32,1)).astype(np.int32)
        info2 = info[:,1].reshape((self.batch_size,1)).astype(np.int32)
        label_info = np.hstack((info1,info2))
        feed_dict = {self.x:x, 
                     self.label:label, 
                     self.label_info:label_info}
        self.sess.run(self.train_step,feed_dict=feed_dict)
    
class Environment(object):
    def __init__(self):
        self._init_env()
        
        self.last_lives = 5                                                    #specify total lives of agent . It varies depending on different games.
        
    def _init_env(self):
        self.env = gym.make('Breakout-v0')
    
    def reset(self):
        self.last_lives = 5
        return self.env.reset()
    
    def trigger_emulator(self,act):
        """This method will trigger emulator and return reward and next state
        by a tuple.And also when you call this method it will render a window
        to show you result where action has been performed."""
        next_state,reward,t_flag,info = self.env.step(act)
        if info['ale.lives'] < self.last_lives:
            reward = info['ale.lives'] - self.last_lives
            self.last_lives = info['ale.lives']
        self.env.render()
        return reward,next_state,t_flag
                                                        
class Learning(object):
    def __init__(self,episode=1000,timestep=1000,capacity=300,gamma=0.8,learning_rate=0.03,batch_size=32,step_size=4):
        self._init_training_parameters(episode,timestep,capacity,gamma,learning_rate,
                                       batch_size,step_size)
        self._init_tf_session()
    
    def _preprocess_seq(self,seq):                                             #ToDo:1:need to check the logic under the whole control flow (Done)
        ret = seq.copy()                                                       #     2:need to add other operations to preprocess image
        ret.list_states = []
        for n in range(len(seq.list_states)) :    
            state = seq.list_states[n] 
            img = Image.fromarray(np.uint8(state))
            img = img.convert("L")
            img = img.crop((38,126,122,210))                                   #crop to size of (84,84) as paper said
            state = np.array(img).astype(np.uint8)
            ret.list_states.append(state)
        return ret
        
    def _init_tf_session(self):
        tf.reset_default_graph()
        self.learningSession = tf.Session()
    
    def _init_training_parameters(self,episode_num,timesteps,capacity,gamma,learning_rate,batch_size,step_size):
        self.episode_num = episode_num
        self.timesteps = timesteps
        self.capacity = capacity
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.step_size = step_size
    
    def train(self):
        
        replaymem = ReplayMemory(self.capacity)
        dqn = DQN(self.learningSession,self.gamma,self.lr,self.batch_size)
        env = Environment()
        loginfo = "Now in episode {epi}; timestep {time}."
        
        for ep in range(self.episode_num):
            seq = StateSequence(env)
            seq.reset()
            for ts in range(self.timesteps):
                print(loginfo.format(epi=ep,time=ts))

                seq_ = self._preprocess_seq(seq)
                act = dqn.select_action(seq_)
                reward,next_state,Done = env.trigger_emulator(act)
                next_seq = seq.expand(act,next_state,Done)
                next_seq_ = self._preprocess_seq(next_seq)
                seq.update(act,next_state,Done)
                transition_tuple = (seq_,act,reward,next_seq_)
                replaymem.store(transition_tuple)
                
                if (ep or ts > self.batch_size) and not ts % self.step_size:
                    batch_transition = replaymem.sample(self.batch_size)
                    dqn.optimize(batch_transition)
                if Done: 
                    print("game is over! next episode...")
                    break
                
        self.learningSession.close()

if __name__ == '__main__':
    dqn_learning = Learning()
    dqn_learning.train()