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
        only transition tuple that is of s_ list whose length larger than 3 will 
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
        self.training_params_ls = []
        self.evaluating_params_ls = []
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self._probobility_space = np.linspace(1,0.1,1000000)
        self.total_step = 0
        self.copy_ops = []
        self._init_graph()
        
    def _init_graph(self):
        self._construct()
        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./log",self.sess.graph)
        writer.close()
        
    def _flatten(self,layers):
        layers = tf.contrib.layers.flatten(layers)
        return layers
    
    def _weights(self,shape,name):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial,name=name)
    
    def _bias(self,shape,name):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial,name=name)
        
    def _main_body(self,used_for_train):
        with tf.name_scope("conv_1"):
            w_h1 = self._weights([8,8,4,16],'w_h1')
            b_h1 = self._bias([16],'b_h1')
            h1_conv = tf.nn.conv2d(self.x,w_h1,[1,4,4,1],'SAME') + b_h1
            h1 = tf.nn.relu(h1_conv)
        with tf.name_scope("conv_2"):
            w_h2 = self._weights([4,4,16,32],'w_h2')
            b_h2 = self._bias([32],'b_h2')
            h2_conv =tf.nn.conv2d(h1,w_h2,[1,2,2,1],'SAME') + b_h2
            h2 = tf.nn.relu(h2_conv)
        with tf.name_scope("fc"):
            h2_flat = self._flatten(h2)
            w_fc = self._weights([h2_flat.shape[1:].num_elements(),256],'w_fc')
            b_fc = self._bias([256],'b_fc')
            fc = tf.nn.relu(tf.matmul(h2_flat,w_fc) + b_fc)
        with tf.name_scope("output"):                                          #self.output has shape of (1,4)
            w_out = self._weights([256,4],'w_out')
            b_out = self._bias([4],'b_out')
            if used_for_train:
                self.output = tf.matmul(fc,w_out) + b_out
                self.training_params_ls.extend([w_h1,b_h1,w_h2,b_h2,w_fc,b_fc,w_out,b_out])
            else:
                self.output_eval = tf.matmul(fc,w_out) + b_out
                self.evaluating_params_ls.extend([w_h1,b_h1,w_h2,b_h2,w_fc,b_fc,w_out,b_out])
        
    def _construct(self):
        """Define what the graph of DQN is.This method was called once instantiate
        a DQN object"""
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,210,160,4])               
        self.label_info = tf.placeholder(tf.int32,None)
        self.label = tf.placeholder(tf.float32,None)
        with tf.name_scope("training_graph"):
            self._main_body(True)                   
            with tf.name_scope("filter_output"):
                self.filtered_output = tf.gather_nd(self.output,indices=self.label_info)
            with tf.name_scope("loss"):
                loss = tf.losses.mean_squared_error(labels=self.label,
                                                    predictions=self.filtered_output)
            with tf.name_scope("optimizer"):
                self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
        with tf.name_scope("evaluating_graph"):
            self._main_body(False)
        with tf.name_scope("copy_op"):
            for t,s in zip(self.evaluating_params_ls,self.training_params_ls):
                self.copy_ops.append(tf.assign(t,s))
                    
    def _compute_label(self,batch_transition):
        ret = []
        for piece in batch_transition:
            if piece[3].list_tflag[-1]:
                ret.append([piece[2],piece[1]])
            else:
                x = np.array(piece[3].list_states[:]).reshape((1,210,160,4))
                feed_dict = {self.x:x}
                max_act_Q_value = np.max(self.sess.run(self.output_eval,feed_dict=feed_dict))
                y = piece[2] + self.gamma * max_act_Q_value
                ret.append([y,piece[1]])
        return np.array(ret).reshape((len(ret),2))
    
    def _make_input_batch_optimize(self,batch_transition):
        ret = []
        for piece in batch_transition:
            ret.append(piece[0].list_states[:])
        return np.array(ret).reshape((self.batch_size,210,160,4))
    
    def _make_input_batch_select_action(self,seq):
        return np.array(seq.list_states[-4:]).reshape((1,210,160,4))
        
    def _backup(self):
        for copy_op in self.copy_ops:
            self.sess.run(copy_op)
    
    def select_action(self,seq,frame):                                         #ToDo:implement other branch of selection strategy,which is random selection (Done)
        if len(seq.list_states) > 3:
            p = (self._probobility_space[frame] if frame <= 999999 else 0.1)
            act_randomly = np.random.binomial(1,p)
            if act_randomly: 
                return random.randrange(0,4)
            else:
                inputs = self._make_input_batch_select_action(seq)
                feed_dict = {self.x:inputs}
                return np.argmax(self.sess.run(self.output,feed_dict=feed_dict))   
        else:
            return random.randrange(0,4)
    @timing
    def optimize(self,batch_transition):                                    
        """update network's parameters"""
        x = self._make_input_batch_optimize(batch_transition)
        info = self._compute_label(batch_transition)
        self._backup()                                                         #once used evaluating graph , training grpah should be backup to evaluating graph
        label = info[:,0].reshape((info[:,0].size,1))
        info1 = np.array(range(32)).reshape((32,1)).astype(np.int32)
        info2 = info[:,1].reshape((self.batch_size,1)).astype(np.int32)
        label_info = np.hstack((info1,info2))
        feed_dict = {self.x:x, 
                     self.label:label, 
                     self.label_info:label_info}
        self.sess.run(self.train_step,feed_dict=feed_dict)
        self.total_step += 1
    
class Environment(object):
    def __init__(self):
        self._init_env()
        self.total_frames = 0
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
        self.total_frames += 1
        if info['ale.lives'] < self.last_lives:
            reward = info['ale.lives'] - self.last_lives
            self.last_lives = info['ale.lives']
        self.env.render()
        return reward,next_state,t_flag
                                                        
class Learning(object):
    def __init__(self,episode=60000,timestep=300000,capacity=2000,gamma=0.7,learning_rate=0.003,batch_size=32,step_size=4,cp_step=500):
        self._init_training_parameters(episode,timestep,capacity,gamma,learning_rate,
                                       batch_size,step_size,cp_step)
        self._init_tf_session()
    
    def _preprocess_seq(self,seq):                                             #ToDo:1:need to check the logic under the whole control flow (Done)
        ret = seq.copy()                                                       #     2:need to add other operations to preprocess image (Done)
        ret.list_states = []
        for n in range(len(seq.list_states)) :    
            state = seq.list_states[n] 
            img = Image.fromarray(np.uint8(state))
            img = img.convert("L")
            state = np.array(img).astype(np.uint8)
            state = state / 255                                                #rescale image into range 0 ~ 1
            ret.list_states.append(state)
        return ret
        
    def _init_tf_session(self):
        tf.reset_default_graph()
        self.learningSession = tf.Session()
    
    def _init_training_parameters(self,episode_num,timesteps,capacity,gamma,learning_rate,batch_size,step_size,cp_step):
        self.episode_num = episode_num
        self.timesteps = timesteps
        self.capacity = capacity
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.step_size = step_size
        self.cp_step = cp_step
        
    def _init_saver(self):
        self.saver = tf.train.Saver()
        
    def _exit_training(self):
        self.learningSession.close()
        self.env.close()
        
    def _model_checkpoint(self):
        self.saver.save(self.learningSession,'./modelcp/dqn_model')
    
    def train(self):
        
        self.replaymem = ReplayMemory(self.capacity)
        self.dqn = DQN(self.learningSession,self.gamma,self.lr,self.batch_size)
        self.env = Environment()
        initial_loginfo = "Now in episode {epi}; timestep {time}."
        ending_loginfo = "Reward is {rwd}; current frame:{frm}"
        
        self._init_saver()
        for ep in range(self.episode_num):
            seq = StateSequence(self.env)
            seq.reset()
            act = 0
            for ts in range(self.timesteps):
                print(initial_loginfo.format(epi=ep,time=ts))
                seq_ = self._preprocess_seq(seq)
                if not (ts % self.step_size):
                    act = self.dqn.select_action(seq_,self.env.total_frames)   #skip step_size frames as paper said
                reward,next_state,Done = self.env.trigger_emulator(act)
                next_seq = seq.expand(act,next_state,Done)
                next_seq_ = self._preprocess_seq(next_seq)
                seq.update(act,next_state,Done)
                transition_tuple = (seq_,act,reward,next_seq_)
                self.replaymem.store(transition_tuple)
                if (self.replaymem.current_capacity > self.batch_size):
                    batch_transition = self.replaymem.sample(self.batch_size)
                    self.dqn.optimize(batch_transition)
                print(ending_loginfo.format(rwd=reward,frm=self.env.total_frames))
                if Done: break
            if (not ep % self.cp_step) or (not ep): self._model_checkpoint()
                
        self._exit_training()

if __name__ == '__main__':
    dqn_learning = Learning()
    dqn_learning.train()