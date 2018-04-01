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
        def expand_(ls,ele):
            if len(ls) > 3: ls.pop(0)
            ls.append(ele)
        next_action_list = self.list_actions.copy()
        next_state_list = self.list_states.copy()
        next_tflag_list = self.list_tflag.copy() 
        list(map(expand_,[next_action_list,next_state_list,next_tflag_list], \
                 [act,next_state,t_flag]))
        next_seq = StateSequence(self.env)
        next_seq.list_actions = next_action_list
        next_seq.list_states = next_state_list
        next_seq.list_tflag = next_tflag_list
        return next_seq
    
    def update(self,act,next_state,t_flag):
        def pop_if_overflow(ls):
            if len(ls) > 3: ls.pop(0)
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
        
    def save(self):
        print("saving replay memory...")
        np.array(self.s_).dump('./modelcp/replaymem_s_.npy')
        np.array(self.next_s_).dump('./modelcp/replaymem_next_s_.npy')
        np.array(self.a).dump('./modelcp/replaymem_a.npy')
        np.array(self.r).dump('./modelcp/replaymem_r.npy')
        print("replay memory saved.")
    
    def load(self):
        print("loading replay memory...")
        self.s_ = list(np.load('./modelcp/replaymem_s_.npy'))
        self.next_s_ = list(np.load('./modelcp/replaymem_next_s_.npy'))
        self.a = list(np.load('./modelcp/replaymem_a.npy'))
        self.r = list(np.load('./modelcp/replaymem_r.npy'))
        print("Done.")
    
class DQN(object):
    def __init__(self,sess,gamma,lr,batch_size,from_scratch=True):
        self.sess = sess
        self.training_params_ls = []
        self.evaluating_params_ls = []
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self._probobility_space = np.linspace(1,0.1,1000000)
        self.p = 1
        self.total_step = 0
        self.copy_ops = []
        self._init_graph(from_scratch)
        
    def _init_graph(self,from_scratch):
        if from_scratch:
            self._construct()
            self.sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("./log",self.sess.graph)
            writer.close()
        else:
            saver = tf.train.import_meta_graph('./modelcp/dqn_model.meta')
            saver.restore(self.sess,tf.train.latest_checkpoint('./modelcp'))
            graph = tf.get_default_graph()
            self.x = graph.get_tensor_by_name("x:0")
            self.label = graph.get_tensor_by_name("label:0")
            self.label_info = graph.get_tensor_by_name("label_info:0")
            self.output = graph.get_tensor_by_name("training_graph/output/output:0")
            self.output_eval = graph.get_tensor_by_name("evaluating_graph/output/output_eval:0")
            self.train_step = graph.get_operation_by_name("training_graph/optimizer/train_step")
            self.copy_ops.append(graph.get_operation_by_name("copy_op/Assign"))
            for n in range(1,8):
                self.copy_ops.append(graph.get_operation_by_name("copy_op/Assign_"+str(n)))

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
                self.output = tf.add(tf.matmul(fc,w_out) , b_out,name="output")
                self.training_params_ls.extend([w_h1,b_h1,w_h2,b_h2,w_fc,b_fc,w_out,b_out])
            else:
                self.output_eval = tf.add(tf.matmul(fc,w_out) , b_out,name="output_eval")
                self.evaluating_params_ls.extend([w_h1,b_h1,w_h2,b_h2,w_fc,b_fc,w_out,b_out])

    def _construct(self):
        """Define what the graph of DQN is.This method was called once instantiate
        a DQN object"""
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,84,84,4],name='x')               
        self.label_info = tf.placeholder(tf.int32,None,name='label_info')
        self.label = tf.placeholder(tf.float32,None,name='label')
        with tf.name_scope("training_graph"):
            self._main_body(True)                   
            with tf.name_scope("filter_output"):
                self.filtered_output = tf.gather_nd(self.output,indices=self.label_info)
            with tf.name_scope("loss"):
                loss = tf.losses.mean_squared_error(labels=self.label,
                                                    predictions=self.filtered_output)
            with tf.name_scope("optimizer"):
                self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(loss,name="train_step")
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
                x = np.array(piece[3].list_states[:]).reshape((1,84,84,4))
                feed_dict = {self.x:x}
                max_act_Q_value = np.max(self.sess.run(self.output_eval,feed_dict=feed_dict))
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
        
    def _backup(self):
        for copy_op in self.copy_ops:
            self.sess.run(copy_op)
    
    def select_action(self,seq,step):                                         #ToDo:implement other branch of selection strategy,which is random selection (Done)
        if len(seq.list_states) > 3:
            self.p = (self._probobility_space[step] if step <= 999999 else 0.1)
            act_randomly = np.random.binomial(1,self.p)
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
            reward = -1.0
            self.last_lives = info['ale.lives']
        self.env.render()
        return reward,next_state,t_flag
                                                        
class Learning(object):
    def __init__(self,epoch=200,epoch_size=50000,capacity=4000,gamma=0.7,learning_rate=0.03,batch_size=32,step_size=4,ckpt_step=50000,from_scratch=True):
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.capacity = capacity
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.step_size = step_size
        self.ckpt_step = ckpt_step
        self.from_scratch = from_scratch
        
        self._init_tf_session()
    
    @property
    def epoch_info(self):
        """return (epoch,step)"""
        return self.dqn.total_step//self.epoch_size,self.dqn.total_step%self.epoch_size
    
    def _preprocess_seq(self,seq):                                             #ToDo:1:need to check the logic under the whole control flow (Done)
        ret = seq.copy()                                                       #     2:need to add other operations to preprocess image (Done)
        ret.list_states = []
        for n in range(len(seq.list_states)) :    
            state = seq.list_states[n]
            img = Image.fromarray(np.uint8(state))
            img = img.convert("L")
            img = img.crop((8,33,152,195))
            img = img.resize((84,84))
            state = np.array(img).astype(np.uint8)
            state = state / 255                                                #rescale image into range 0 ~ 1
            ret.list_states.append(state)
        return ret
        
    def _init_tf_session(self):
        tf.reset_default_graph()
        self.learningSession = tf.Session()
        
    def _init_saver(self):
        self.saver = tf.train.Saver()
        
    def _restore(self):
        with open('./modelcp/record','r') as f:
            self.dqn.total_step = int(f.read())
        self.replaymem.load()
        
    def _exit_training(self):
        self.learningSession.close()
        self.env.close()
        
    def _model_checkpoint(self):
        self.saver.save(self.learningSession,'./modelcp/dqn_model')
        with open('./modelcp/record','w+') as f:
            f.write(str(self.dqn.total_step))
        self.replaymem.save()
    
    def train(self):
        self.replaymem = ReplayMemory(self.capacity)
        self.dqn = DQN(self.learningSession,self.gamma,self.lr,self.batch_size,self.from_scratch)
        self.env = Environment()
        initial_loginfo = """Epoch {epo}; step {stp}
Episode {epi}; Timestep {time}
Act randomly with {p}"""
        ending_loginfo = """Reward {rwd}\nTotal Step {opt}"""
        
        self._init_saver()
        if not self.from_scratch : self._restore()
        for ep in range(int(1e7)):
            if self.dqn.total_step > self.epoch * self.epoch_size: 
                print("training completed.")                
                break          
            seq = StateSequence(self.env)
            seq.reset()
            act = 0
            for ts in range(int(1e7)):
                print(initial_loginfo.format(epo=self.epoch_info[0],
                                             stp=self.epoch_info[1],
                                             epi=ep,time=ts,
                                             p=self.dqn.p))
                seq_ = self._preprocess_seq(seq)
                if not (ts % self.step_size):
                    act = self.dqn.select_action(seq_,self.dqn.total_step)     #skip step_size frames as paper said
                reward,next_state,Done = self.env.trigger_emulator(act)
                next_seq = seq.expand(act,next_state,Done)
                next_seq_ = self._preprocess_seq(next_seq)
                seq.update(act,next_state,Done)
                transition_tuple = (seq_,act,reward,next_seq_)
                self.replaymem.store(transition_tuple)
                if (self.replaymem.current_capacity > self.batch_size):
                    batch_transition = self.replaymem.sample(self.batch_size)
                    self.dqn.optimize(batch_transition)
                print(ending_loginfo.format(rwd=reward,opt=self.dqn.total_step))
                if Done: break
                if ((not self.dqn.total_step % self.ckpt_step) and (self.dqn.total_step)):
                    self._model_checkpoint()
                
        self._exit_training()

if __name__ == '__main__':
    dqn_learning = Learning(from_scratch=False)
    dqn_learning.train()