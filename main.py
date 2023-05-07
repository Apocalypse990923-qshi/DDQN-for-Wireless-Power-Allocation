from qlearning import *
from qlearning2 import *
from dlmodel import *
from duelingDQNmodel import *
from utils import *
from sklearn.model_selection import train_test_split

def draw_capacity(cap, count, labels=['Water-filling', 'Deep Q-Learning', 'Q-learning', 'Deep Learning', 'Dueling DQN', 'Dueling DL','Q-learning2'], c=['r','b','g','k','y','cyan','magenta']):
        for i in range(len(cap)):
            plt.plot(range(count), cap[i], c[i], label=labels[i])
        plt.xlabel('iterations')
        plt.ylabel('Spectral-Efficiency(bps/Hz)')
#        plt.plot(range(count), cap[1], 'b', label=labels[1])
#        if cap.shape[0] == 3:
#            plt.plot(range(count), cap[2], 'g', label=labels[2])
        plt.legend()
        # plt.show()
        tname = str(int(time.time()))
        filename = 'QFNet_' + tname + '_.png'
        plt.savefig(Tools.results + '/' + filename)
        with open(Tools.results + '/' + tname + '.txt', 'w') as fw:
            fw.write(str(cap.flatten()))
        plt.close()

def draw(labels=['Water-filling', 'Deep Q-Learning', 'Q-learning', 'Deep Learning', 'Dueling DQN', 'Dueling DL','Q-learning2'], c=['r','b','g','k','y','cyan','magenta']):
        cap = np.zeros((6, 20))
        with open(Tools.results + '/result.txt', 'r') as fr:
            data = fr.read().strip().replace('[', '').replace(']', '').split()
        cap = np.array(data, dtype=float).reshape(6, 20)
        print(cap)
        for i in range(len(cap)-1):
            plt.plot(range(20), cap[i], c[i], label=labels[i])
        plt.xlabel('iterations')
        plt.ylabel('Spectral-Efficiency(bps/Hz)')
#        plt.plot(range(count), cap[1], 'b', label=labels[1])
#        if cap.shape[0] == 3:
#            plt.plot(range(count), cap[2], 'g', label=labels[2])
        plt.legend(loc='upper right')
        #plt.show()
        filename = 'result1_without_DuelDL.png'
        plt.savefig(Tools.results + '/' + filename)
        plt.close()

def evaluate(ql, dl, ql2, duel_dl, epoch=10):
    '''该函数用来判断模型的质量'''
    net = Network()
    net.start()
    model_best = dl.load_best_model()
    ql_only_best = ql.load_qlmodel(Tools.ql_model)
    qf_best = ql.load_qlmodel(Tools.ql_model_pro)

    #net2 = Network()
    #net2.start()
    duel_model_best = duel_dl.load_best_model()
    duel_ql_only_best = ql2.load_qlmodel(Tools.duel_ql_model)
    duel_qf_best = ql2.load_qlmodel(Tools.duel_ql_model_pro)

    #cap = np.zeros((7, epoch))
    cap = np.zeros((6, epoch))
    for i in range(epoch):
        net.update()
        epoch_data = Tools.get_epoch_data(net)
        print('Calculating Normal Situation')
        cap[0, i] = net.compute_capacity(net.power_subcarrier, net.BS_power)

        print('Calculating QFNet')
        outs_best = dl.predict(model_best, epoch_data)
        outs_best = np.reshape(outs_best, (net.expectBS, net.subcarrier))
        qfnet_capacity = net.compute_capacity(
            outs_best + net.power_subcarrier,
            net.BS_power + np.sum(outs_best, axis=1)
        )
        qfouts = ql.ql_evaluate(qf_best, net, iteration=1)
        qfouts = np.reshape(qfouts, (net.expectBS, net.subcarrier))
        qfouts_capacity = net.compute_capacity(
            qfouts + net.power_subcarrier,
            net.BS_power + np.sum(qfouts, axis=1)
        )
        cap[1, i] = qfouts_capacity
        print('Calculating q1')
        qlouts = ql.ql_evaluate(ql_only_best, net, iteration=1)
        qlouts = np.reshape(qlouts, (net.expectBS, net.subcarrier))
        qlouts_capacity = net.compute_capacity(
            qlouts + net.power_subcarrier,
            net.BS_power + np.sum(qlouts, axis=1)
        )
        cap[2, i] = qlouts_capacity
        cap[3, i] = qfnet_capacity
        print('epoch: ', i, 'normal: ', cap[0, i], 'qfouts: ', cap[1, i], 'ql: ', cap[2, i], 'qfnet: ', cap[3, i])

        #Duel
        '''
        net2.update()
        epoch_data2 = Tools.get_epoch_data(net2)
        print('Calculating Duel DQN')
        outs_best = duel_dl.predict(duel_model_best, epoch_data2)
        outs_best = np.reshape(outs_best, (net2.expectBS, net2.subcarrier))
        qfnet_capacity = net2.compute_capacity(
            outs_best + net2.power_subcarrier,
            net2.BS_power + np.sum(outs_best, axis=1)
        )
        qfouts = ql2.ql_evaluate(duel_qf_best, net2, iteration=1)
        qfouts = np.reshape(qfouts, (net2.expectBS, net2.subcarrier))
        qfouts_capacity = net2.compute_capacity(
            qfouts + net2.power_subcarrier,
            net2.BS_power + np.sum(qfouts, axis=1)
        )
        cap[4, i] = qfouts_capacity
        print('Calculating q2')
        qlouts = ql2.ql_evaluate(duel_ql_only_best, net2, iteration=1)
        qlouts = np.reshape(qlouts, (net2.expectBS, net2.subcarrier))
        qlouts_capacity = net2.compute_capacity(
            qlouts + net2.power_subcarrier,
            net2.BS_power + np.sum(qlouts, axis=1)
        )
        cap[5, i] = qlouts_capacity
        #cap[6, i] = qfnet_capacity
        #print('epoch: ', i, 'qfouts: ', cap[4, i], 'ql: ', cap[5, i], 'qfnet: ', cap[6, i])
        print('epoch: ', i, 'qfouts: ', cap[4, i], 'ql: ', cap[5, i])
        '''
        print('Calculating Duel DQN')
        outs_best = duel_dl.predict(duel_model_best, epoch_data)
        outs_best = np.reshape(outs_best, (net.expectBS, net.subcarrier))
        qfnet_capacity = net.compute_capacity(
            outs_best + net.power_subcarrier,
            net.BS_power + np.sum(outs_best, axis=1)
        )
        qfouts = ql2.ql_evaluate(duel_qf_best, net, iteration=1)
        qfouts = np.reshape(qfouts, (net.expectBS, net.subcarrier))
        qfouts_capacity = net.compute_capacity(
            qfouts + net.power_subcarrier,
            net.BS_power + np.sum(qfouts, axis=1)
        )
        cap[4, i] = qfouts_capacity
        cap[5, i] = qfnet_capacity
        #print('Calculating q2')
        qlouts = ql2.ql_evaluate(duel_ql_only_best, net, iteration=1)
        qlouts = np.reshape(qlouts, (net.expectBS, net.subcarrier))
        qlouts_capacity = net.compute_capacity(
            qlouts + net.power_subcarrier,
            net.BS_power + np.sum(qlouts, axis=1)
        )
        #cap[5, i] = qlouts_capacity
        #print('epoch: ', i, 'qfouts: ', cap[4, i], 'ql: ', cap[5, i], 'qfnet: ', cap[6, i])
        print('epoch: ', i, 'duel qfouts: ', cap[4, i], 'duel dnn: ', cap[5, i])
    #dl.draw_capacity(cap, epoch)
    draw_capacity(cap, epoch)


def create_only_q():
    #net = Network()
    #net.start()
    ql = Qlearning()
    if len(os.listdir(Tools.ql_model)) == 0:
        print('Train from beginning')
        ql.start('0', epoch=5, iteration=16, only_q=True)

def create_only_q_duel():
    #net = Network()
    #net.start()
    ql2 = Qlearning2()
    if len(os.listdir(Tools.duel_ql_model)) == 0:
        print('Train from beginning')
        ql2.start('0', epoch=5, iteration=16, only_q=True)

def draw_convergence():
    only_c = open(Tools.qfnet_data + '/' + '0' + '_only_c.txt', 'r')
    only_c = np.array(only_c.read().split(',')[:-1])
    only_c = str_map_float(only_c)
    dql = open(Tools.qfnet_data + '/' + '1' + '_dqn_c.txt', 'r')
    dql = np.array(dql.read().split(',')[:-1])
    dql = str_map_float(dql)
    n = range(min(len(dql), len(only_c)))
    only_c[0:] = np.array(only_c)[0:]
    dql[0:] = np.array(dql)[0:]
    plt.plot(n, only_c[0:len(n)], 'b', label='DQFC Q-Learning')
    plt.plot(n, dql[0:len(n)], 'r', label='Q-learning')
    plt.ylabel('Spectral-Efficiency(bps/Hz)')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig(Tools.qf_image + '/' + 'convergence.png')
    plt.close()

    only_c = open(Tools.qfnet_data + '/duel_' + '0' + '_only_c.txt', 'r')
    only_c = np.array(only_c.read().split(',')[:-1])
    only_c = str_map_float(only_c)
    dql = open(Tools.qfnet_data + '/duel_' + '1' + '_dqn_c.txt', 'r')
    dql = np.array(dql.read().split(',')[:-1])
    dql = str_map_float(dql)
    n = range(min(len(dql), len(only_c)))
    only_c[50:] = np.array(only_c)[50:]
    dql[50:] = np.array(dql)[50:]
    plt.plot(n, only_c[0:len(n)], 'r', label='Q-learning2')
    plt.plot(n, dql[0:len(n)], 'b', label='Dueling DQN')
    plt.ylabel('Spectral-Efficiency(bps/Hz)')
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig(Tools.duel_qf_image + '/' + 'convergence2.png')
    plt.close()

    only_c = open(Tools.qfnet_data + '/duel_' + '0' + '_only_c.txt', 'r')
    only_c = np.array(only_c.read().split(',')[:-1])
    only_c = str_map_float(only_c)
    dql = open(Tools.qfnet_data + '/' + '1' + '_dqn_c.txt', 'r')
    dql = np.array(dql.read().split(',')[:-1])
    dql = str_map_float(dql)
    duel_dql = open(Tools.qfnet_data + '/duel_' + '1' + '_dqn_c.txt', 'r')
    duel_dql = np.array(duel_dql.read().split(',')[:-1])
    duel_dql = str_map_float(duel_dql)
    n = range(min(len(dql), len(duel_dql)))
    dql[0:] = np.array(dql)[0:]
    duel_dql[0:] = np.array(duel_dql)[0:]
    plt.plot(n, only_c[0:len(n)], 'g', label='Q-Learning Only')
    plt.plot(n, dql[0:len(n)], 'b', label='DQFC Q-Learning')
    plt.plot(n, duel_dql[0:len(n)], 'r', label='Dueling DQFC Q-Learning')
    plt.ylabel('Spectral-Efficiency(bps/Hz)')
    plt.xlabel('iterations')
    plt.title('Convergence')
    plt.legend()
    plt.savefig(Tools.qfnet_data + '/' + 'convergence_combine.png')
    plt.close()


def start_net():
    #net = Network()
    #net.start()
    #net2 = Network()
    #net2.start()
    qfnet = DLModel()
    duel_qfnet = Duel_DLModel()
    qlearning = Qlearning()
    qlearning2 = Qlearning2()
    
    for i in range(1,5):
        
        model_name = str(i)
        print('Geneate the ' + model_name + 'th model')
        
        qlearning.start(model_name, epoch=5, iteration=16, only_q=False)
        model = qfnet.build_model()
        x_train, y_train = qfnet.load_data(model_name)
        train_data, _, train_lab, _ = train_test_split(x_train, y_train, test_size=0.2, random_state=36)
        qfnet.train(model, train_data, train_lab, epoch=5)
        # qfnet.train(model, x_train, y_train, epoch=50)
        qfnet.evaluate(model, epoch=5)
        
        qlearning2.start(model_name, epoch=5, iteration=16, only_q=False)
        duel_model = duel_qfnet.build_model()
        x_train, y_train = duel_qfnet.load_data(model_name)
        train_data, _, train_lab, _ = train_test_split(x_train, y_train, test_size=0.2, random_state=36)
        duel_qfnet.train(duel_model, train_data, train_lab, epoch=5)
        duel_qfnet.evaluate(duel_model, epoch=5)
        
        evaluate(qlearning, qfnet, qlearning2, duel_qfnet, epoch=20)
        # qlearning.remove_data()  # 删除训练数据
    
    #qfnet.model_summary()
    #qfnet.examples(epoch=20)
    #duel_qfnet.model_summary()
    #duel_qfnet.examples(epoch=20)



if __name__ == '__main__':

    #Tools.create_dirs()
    #create_only_q()
    #create_only_q_duel()
    #start_net()
    draw_convergence()
    #draw()
    '''
    net = Network()
    net.start()
    fig = plt.figure(1)
    window1 = fig.add_subplot(111)
    window1.scatter(net.BS_location[:, 0], net.BS_location[:, 1],marker='^', c='r')
    #window2 = fig.add_subplot(111)
    window1.scatter(net.UE_location[:, 0], net.UE_location[:, 1], marker='*', c='b')
    plt.title("BS-UE Topology")
    plt.legend(['BS','UE'])
    #plt.show()
    plt.savefig(Tools.qfnet_data + '/' + 'topology.png')
    '''
