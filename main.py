import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
import torch.optim as optim
from models import *  # PyTorch 모델 정의가 포함된 가정
from utils import *  # PyTorch에 맞게 수정된 유틸리티 함수 가정
def fit_GAN(run, g_model, d_model, c_model, gan_model, n_samples, n_classes, X_sup, y_sup, dataset, n_epochs, n_batch, latent_dim=100):
    tst_history = []
    X_tra, y_tra, X_tst, y_tst = dataset
    bat_per_epo = int(X_tra.shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)

    # Optimizers
    optimizer_g = optim.Adam(g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_c = optim.Adam(c_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for i in range(n_steps):
        # update discriminator (c)
        Xsup_real, ysup_real = generate_real_samples([X_sup, y_sup], half_batch)
        Xsup_real, ysup_real = torch.tensor(Xsup_real).float(), torch.tensor(ysup_real).float()
        c_model.zero_grad()
        y_pred = c_model(Xsup_real)
        c_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, ysup_real)
        c_loss.backward()
        optimizer_c.step()

        # update discriminator (d)
        X_real, y_real = generate_real_samples((X_tra, y_tra), half_batch)
        X_real, y_real = torch.tensor(X_real).float(), torch.tensor(y_real).float()
        d_model.zero_grad()
        y_pred_real = d_model(X_real)
        d_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(y_pred_real, y_real)
        d_loss_real.backward()

        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        X_fake, y_fake = torch.tensor(X_fake).float(), torch.tensor(y_fake).float()
        y_pred_fake = d_model(X_fake)
        d_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(y_pred_fake, y_fake)
        d_loss_fake.backward()
        optimizer_d.step()

        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), torch.ones((n_batch, 1))
        gan_model.zero_grad()
        y_pred_gan = gan_model(X_gan)
        g_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred_gan, y_gan)
        g_loss.backward()
        optimizer_g.step()
        # summarize loss on this batch
        print('>%d/%d/%d, c[%.3f], d[%.3f,%.3f], g[%.3f]' % (run+1, i+1, n_steps, c_loss.item(), d_loss_real.item(), d_loss_fake.item(), g_loss.item()))
        # test after a epoch
        if (i+1) % (bat_per_epo * 1) == 0:
            _, _acc = c_model.evaluate(X_tst, y_tst, verbose=0)
            tst_history.append(_acc)

    return tst_history

def select_supervised_samples(X, Y, n_samples, n_classes):  # X=데이터셋의 X값, Y=데이터셋의 Y값 ＜＜＜ (X_tra, y_tra, n_samples[j],  n_classes)        
    X_list, Y_list = list(), list()  # 새 리스트 변수 생성
    n_per_class = int(n_samples/n_classes)  # 클래스 당 샘플 갯수

    for i in range(n_classes):  # y는 클래스값!! 즉 레이블값=클래스,
        X_with_class = X[Y==i]  # 데이터셋 X에서 Y가 i인 인스턴스들만 선별하여 새로운 데이터셋 X_with_class를 만듭니다.
        ix = np.random.randint(0, len(X_with_class), n_per_class)  # np.random.randint (x, y, size) 범위안에 있는 정수 값을 랜덤으로 지정된 배열의 크기만큼 생성한다. 이 때 x와 y값은 범위의 시작과 끝값이며 size는 array의 크기를 의미한다.
        [X_list.append(X_with_class[j]) for j in ix]  # x_list에 클래스가 ix인 x with classv값들을 저장
        [Y_list.append(i) for j in ix]   # 그 클래스들을 저장
    return np.asarray(X_list), np.asarray(Y_list)   # 즉 특정한 클래스값 Y를 가진 데이터 X 만 모으고서, 이들을 X_list Y_list로 출력해내는 기능을 한다.

def generate_real_samples(dataset, n_samples):

    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    Y=np.ones((n_samples, 1))
    return [X, labels], Y 

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)

    return z_input  

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict(z_input)
    y = np.zeros((n_samples, 1))
    return images, y   


def run_exp1():
    #experiment setup
    n_classes = 16
    n_samples = [16] # define the number of labeled samples here
    run_times = 1
    optimizer = optim.Adam
    n_epochs = 100
    n_batch = 128

    #load dataset
    pre_dataset = pickle.load(open('dataset/EXP1.pickle','rb'))
    dataset = data_preproc(np.asarray(pre_dataset))
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []

        # select supervised dataset
        X_sup, y_sup = select_supervised_samples(X_tra, y_tra, n_samples[j],  n_classes)
        for i in range(run_times):
            print('{}/{}'.format(i+1, run_times))
            # change seed for each run
            # torch.manual_seed(run_times)
            torch.manual_seed(i)  # 버그 수정: run_times 대신 i를 사용하여 각 실행마다 다른 시드를 제공
            # define a semi-GAN model
            d_model, c_model = define_discriminator(n_classes, optimizer)
            g_model = define_generator()
            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i ,g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)
            history.append(max(tst_acc))
        best = max (history)
        # save results:
        fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        pickle.dump(history, fh)
        fh.close()
        # save models:
        #torch.save(g_model.state_dict(), 'exp1_result/GAN-g-exp1-{}-{}.pt'.format(n_samples[j], int(best*100)))
        #torch.save(d_model.state_dict(), 'exp1_result/GAN-d-exp1-{}-{}.pt'.format(n_samples[j], int(best*100)))
        #torch.save(c_model.state_dict(), 'exp1_result/GAN-c-exp1-{}-{}.pt'.format(n_samples[j], int(best*100)))


def run_exp2():
    #experiment setup
    n_classes = 14
    n_samples = [14] # define the number of labeled samples here
    run_times = 1
    optimizer = optim.Adam(lr=0.0002, betas=(0.5, 0.999))
    n_epochs = 100
    n_batch = 128

    #load dataset
    dataset = data_preproc(np.asarray(pickle.load(open('dataset/EXP2.pickle','rb'))))
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup, y_sup = select_supervised_samples(X_tra, y_tra, n_samples[j],  n_classes)        
        for i in range(run_times):
            print('{}/{}'.format(i+1, run_times))
            # change seed for each run
            torch.manual_seed(run_times)
            # define a semi-GAN model
            d_model, c_model = define_discriminator(n_classes, optimizer)
            g_model = define_generator()
            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i ,g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

            history.append(max(tst_acc))
            #history.append(tst_acc)
        best = max (history)
        # save results:
        #fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #pickle.dump(history, fh)
        #fh.close()
        # save models:
        #torch.save(g_model.state_dict(), 'exp2_result/GAN-g-exp2-{}-{}.pt'.format(n_samples[j], int(best*100)))
        #torch.save(d_model.state_dict(), 'exp2_result/GAN-d-exp2-{}-{}.pt'.format(n_samples[j], int(best*100)))
        #torch.save(c_model.state_dict(), 'exp2_result/GAN-c-exp2-{}-{}.pt'.format(n_samples[j], int(best*100)))


def run_exp3():
    # experiment setup #Train: 400/loc, 6400 in total
    n_classes = 18
    n_samples = [3600]
    run_times = 1
    optimizer = optim.Adam
    n_epochs = 100
    n_batch = 128

    #load dataset
    dataset1 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r1.pickle','rb'))))
    dataset2 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r2.pickle','rb'))))
    X_tra1, y_tra1, X_tst1, y_tst1 = dataset1
    X_tra2, y_tra2, X_tst2, y_tst2 = dataset2

    # combine the data from r1/r2
    X_tra = np.concatenate((X_tra1, X_tra2))
    y_tra = np.concatenate((y_tra1, y_tra2))
    X_tst = np.concatenate((X_tst1, X_tst2))
    y_tst = np.concatenate((y_tst1, y_tst2))
    dataset = (X_tra, y_tra, X_tst, y_tst)

    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup1, y_sup1 = select_supervised_samples(X_tra1, y_tra1, n_samples[j],  n_classes)
        X_sup2, y_sup2 = select_supervised_samples(X_tra2, y_tra2, n_samples[j],  n_classes)
        X_sup = np.concatenate((X_sup1, X_sup2))
        y_sup = np.concatenate((y_sup1, y_sup2))        
        for i in range(run_times):
            print('{}/{}'.format(i+1, run_times))
            # change seed for each run
            torch.manual_seed(run_times)
            # define a semi-GAN model
            d_model, c_model = define_discriminator(n_classes, optimizer)
            g_model = define_generator()
            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i ,g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

            history.append(max(tst_acc))
            #history.append(tst_acc)
        best = max (history)


        #torch.save(g_model.state_dict(), 'exp3_result/GAN-g-exp1-{}-{}.pt'.format(n_samples[j], int(best*100)))
        #torch.save(d_model.state_dict(), 'exp3_result/GAN-d-exp1-{}-{}.pt'.format(n_samples[j], int(best*100)))
        #torch.save(c_model.state_dict(), 'exp3_result/GAN-c-exp1-{}-{}.pt'.format(n_samples[j], int(best*100)))

        #fh = open('GAN-r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #fh = open('GANr1r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #pickle.dump(history, fh)
        #fh.close()

def run_cnn():
    '''
    Run CNN under different number of supervised samples
    '''

    # experiment setup
    n_classes = 18
    n_samples = [18, 36, 72, 1800, 3600]
    run_times = 10
    optimizer = optim.Adam
    batch_size = 18 
    epochs = 50

    #load dataset
    dataset1 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r1.pickle','rb'))))
    dataset2 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r2.pickle','rb'))))
    X_tra1, y_tra1, X_tst1, y_tst1 = dataset1
    X_tra2, y_tra2, X_tst2, y_tst2 = dataset2 
    X_tra = np.concatenate((X_tra1, X_tra2))
    y_tra = np.concatenate((y_tra1, y_tra2))
    X_tst = np.concatenate((X_tst1, X_tst2))
    y_tst = np.concatenate((y_tst1, y_tst2))
    
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup1, y_sup1 = select_supervised_samples(X_tra1, y_tra1, n_samples[j],  n_classes)
        X_sup2, y_sup2 = select_supervised_samples(X_tra2, y_tra2, n_samples[j],  n_classes)
        X_sup = np.concatenate((X_sup1, X_sup2))
        y_sup = np.concatenate((y_sup1, y_sup2))        

        for i in range(run_times):
            torch.manual_seed(run_times)
            model = CNN (n_classes, optimizer)
            print('{}/{}'.format(i+1, run_times))
            model.fit(X_sup, y_sup, batch_size, epochs, verbose = 1)
            tst_acc = model.evaluate(X_tst, y_tst)[1]
            print("Test Acc = {}".format(tst_acc))
            history.append(tst_acc)
        best = max (history)

        #fh = open('GAN-r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        fh = open('CNNr1r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        pickle.dump(history, fh)
        fh.close()    

if __name__ == '__main__':

    run_exp1()
    # main에서 실행되는 코드를 디버깅하듯 따라가며 이해하기!
