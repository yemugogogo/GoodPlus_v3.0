from sklearn import svm,preprocessing
import numpy as np
from firebase import firebase
import json


if __name__ == "__main__":
    #讀入正確坐姿與錯誤坐姿
    with open('correct399.json', 'r') as f:
        jf_t = json.loads(f.read())
    with open('incorrect356.json', 'r') as f1:
        jf_f = json.loads(f1.read())
    
    #建空的array
    a=np.empty([0,5])
    b=np.empty([0,5])
    a_x=np.empty([0,5])
    a_y=np.empty([0,5])
    b_x=np.empty([0,5])
    b_y=np.empty([0,5])
    test_x=np.empty([0,5])
    test_y=np.empty([0,5])
    data_caculate=np.empty([0,5])
    
    
    
    
    #以檢索方式讀入正確與錯誤坐姿的xy座標
    for i in range(40):
        for j in range(5):
            x=jf_t[i]['keypoints'][j]['position']['x']
            y=jf_t[i]['keypoints'][j]['position']['y']
            a_x=np.append(a_x,x)
            a_y=np.append(a_y,y)
    a_x=preprocessing.scale(a_x)
    a_y=preprocessing.scale(a_y)
    #a=np.concatenate((a_x,a_y),axis=0)
    
    for i in range(200):
            a=np.append(a,a_x[i])
            a=np.append(a,a_y[i])
    
    a=a.reshape(40,10)
    
    for i in range(40):
        for j in range(5):
            x=jf_f[i]['keypoints'][j]['position']['x']
            y=jf_f[i]['keypoints'][j]['position']['y']
            b_x=np.append(b_x,x)
            b_y=np.append(b_y,y)
    b_x=preprocessing.scale(b_x)
    b_y=preprocessing.scale(b_y)
    #b=np.concatenate((b_x,b_y),axis=0)
    
    #這邊是整理成(x1,y1),(x2,y2)...
    #20*5(測資數*5個點)
    for i in range(200):
            b=np.append(b,b_x[i])
            b=np.append(b,b_y[i])
    
    b=b.reshape(40,10)
    #print('b[100]=%s'%b[100])
    #將正確與錯誤坐姿array合併
    X=np.concatenate((a,b),axis=0)
    
    # A B 對應XY的擬合資料
    
    A = ('Y','Y')
    B = ('N','N')
    Y = 20*A+20*B
    
    
    
    cw = {'Y':1,'N':1}
    
    sv = svm.SVC(decision_function_shape='ovr',
                 C=100000,
                 class_weight=cw,
                 kernel='rbf', gamma=2)
     #           kernel='linear')
    
    sv.fit(X, Y) 
    global count
    count=0
    #讀入欲預測的坐姿
    #with open('E:\\Posenet_Points (13).json', 'r') as f2:
    #    test = json.loads(f2.read())
    file = open('../../../user_id.txt', 'r') 
    name=file.read() 
    
    while True:
        test_data=np.empty([0,5])
        testA = np.empty([0,5])
        
        firebase_temp = firebase.FirebaseApplication('https://wellsitting-ef0e5.firebaseio.com/temp/', None)
        #name = firebase_temp.get('name',None)
        
        ref1 = firebase_temp.get(name+'/nose_x',None)
        ref2 = firebase_temp.get(name+'/nose_y',None)
        ref3 = firebase_temp.get(name+'/leftEye_x',None)
        ref4 = firebase_temp.get(name+'/leftEye_y',None)
        ref5 = firebase_temp.get(name+'/rightEye_x',None)
        ref6 = firebase_temp.get(name+'/rightEye_y',None)
        ref7 = firebase_temp.get(name+'/leftEar_x',None)
        ref8 = firebase_temp.get(name+'/leftEar_y',None)
        ref9 = firebase_temp.get(name+'/rightEar_x',None)
        ref10 = firebase_temp.get(name+'/rightEar_y',None)
    	
        
        test_x=np.append(test_x,[ref1,ref3,ref5,ref7,ref9])
        #print("test_x:",test_x)
        test_y=np.append(test_y,[ref2,ref4,ref6,ref8,ref10])
        #print("test_y:",test_y)
        test_x=preprocessing.scale(test_x)
        test_y=preprocessing.scale(test_y)
        data_prepare=np.concatenate((test_x,test_y),axis=0)
        
        for i in range(5):
            testA=np.append(testA,test_x[i])
            testA=np.append(testA,test_y[i])
        
        test_data = testA.reshape(1, 10)
        #用檢索的方式把資料放進b裡面
        
        #for j in range(5):
            #x=test[0]['keypoints'][j]['position']['x']
            #y=test[0]['keypoints'][j]['position']['y']
            #test_x=np.append(test_x,x)
            #test_y=np.append(test_y,y)
            #用來計算的測資
            #data_caculate=np.append(data_caculate,[x,y])
        #print(data_caculate)
        #test_x=preprocessing.scale(test_x)
        #test_y=preprocessing.scale(test_y)
        #data_prepare=np.concatenate((test_x,test_y),axis=0)
        #for i in range(5):
                #testA=np.append(testA,test_x[i])
                #testA=np.append(testA,test_y[i])
                
        data_caculate=[]
        data_caculate=np.append(data_caculate,[ref1,ref2,ref3,ref4,ref5,ref6,ref7,ref8,ref9,ref10])
        #print(data_caculate)
        
        #print(testA)
        #test_data = testA.reshape(1, 10)#每10個數字成一組
        #print(test_data)
        
        
        # ovo: one-against-one, ovr:one-vs-the-rest
        print('\n Predict for the test data(svm) =  ', sv.predict(test_data))
        if(sv.predict(test_data)=='Y'):
            svm=1
        else:
            svm=0
        print("svm:",svm)
        #----------------------------------計算部分--------------------------------------------
        #錯誤訊息參數
        distanceLog = 0 #顯示距離的錯誤訊息
        slopeLog = 0 #顯示斜率的錯誤訊息
        Msg = "錯誤訊息："
        MsgLog = "錯誤細項："
        
        
        #海龍公式-面積(眼鼻構成之三角形)，用以判斷遠近
        length_a=((data_caculate[0]-data_caculate[2])**2+(data_caculate[1]-data_caculate[3])**2)**0.5
        length_b=((data_caculate[2]-data_caculate[4])**2+(data_caculate[3]-data_caculate[5])**2)**0.5
        length_c=((data_caculate[0]-data_caculate[4])**2+(data_caculate[1]-data_caculate[5])**2)**0.5
        s=(length_a+length_b+length_c)/2
        area=(s*(s-length_a)*(s-length_b)*(s-length_c))**0.5
        #print(area)
        
#        if(area > 1812.3617 or area < 575.9319):
        if(area > 1812.3617):
            distance = 0
            if(area > 1812.3617):
                distanceLog = distanceLog + 1
                MsgLog += " 眼鼻-太近 "
#            else:
#                distanceLog = distanceLog - 1
#                MsgLog += " 眼鼻-太遠 "
        else:
            distance = 1
        print("distance:",distance)
        print("eyesArea: ",area)
        
        
        '''
        #海龍公式-面積(眼耳構成之三角形)，用以判斷遠近
        length_a2=((data_caculate[0]-data_caculate[6])**2+(data_caculate[1]-data_caculate[7])**2)**0.5
        length_b2=((data_caculate[6]-data_caculate[8])**2+(data_caculate[7]-data_caculate[9])**2)**0.5
        length_c2=((data_caculate[8]-data_caculate[0])**2+(data_caculate[9]-data_caculate[1])**2)**0.5
        s2=(length_a2+length_b2+length_c2)/2
        area2=(s2*(s2-length_a2)*(s2-length_b2)*(s2-length_c2))**0.5
        print(area2)
        
        if(area2 > 3332.6593 or area2 < 673.0129):
            distance2 = 0
            if(area > 3332.6593):
                distanceLog = distanceLog + 1
                MsgLog += " 耳鼻-太近 "
            else:
                distanceLog = distanceLog - 1
                MsgLog += " 耳鼻-太遠 "
        else:
            distance2 = 1
            
        print("distance2:",distance2)
        print("earsArea: ",area2)
        '''
        
        #眼斜率
        eye_slope_original=(data_caculate[5]-data_caculate[3])/(data_caculate[4]-data_caculate[2])
        eye_slope = abs(eye_slope_original)
        
        if(eye_slope <= 0.1310):
            head_by_eye = 1
        else:
            head_by_eye = 0
            if(eye_slope_original > 0):
                slopeLog = slopeLog + 1
                MsgLog += " 眼斜-太右 "
            elif(eye_slope_original < 0):
                slopeLog = slopeLog - 1
                MsgLog += " 眼斜-太左 "
                
        print("head_by_eye",head_by_eye)
        print("eyesSlope: ",eye_slope)
        
        
        #耳斜率
        ear_slope=abs(data_caculate[9]-data_caculate[7])/(data_caculate[8]-data_caculate[6])
        
        if(ear_slope <= 0.1382):
            head_by_ear = 1
        else:
            head_by_ear = 0
            if(eye_slope_original > 0):
                slopeLog = slopeLog + 1
                MsgLog += " 耳斜-太右 "
            elif(eye_slope_original < 0):
                slopeLog = slopeLog - 1
                MsgLog += " 耳斜-太左 "
                
        print("head_by_ear",head_by_ear)
        print("earsSlope: ",ear_slope)
        
        '''
        #當兩項距離皆為過近，或過遠，則給予負權重
        if(distance == distance2 and distance == 0):
            distance = (-1)
            distance2 = (-1)
            if(distanceLog == 2):
                Msg += " 太近了 "
            elif(distanceLog == -2):
                Msg += " 太遠了 " 
        '''
        #當兩項斜率皆為傾向左邊，或右邊，則給予負權重
        if(head_by_eye == head_by_ear and head_by_eye == 0):
            head_by_eye = (-1)
            head_by_ear = (-1)
            if(slopeLog == 2):
                Msg += " 太右了 "
            elif(slopeLog == -2):
                Msg += " 太左了 " 
        
             
        result = (0.2*svm) + (0.3*distance) + (0.25*head_by_eye) + (0.25*head_by_ear)
        print(result)
        print(Msg)
        print(MsgLog)
        data_test=np.empty([0,1])
        
        if(result>0.6):
            print("Y")
            data_test=np.delete(data_test,[0])
            data_test=np.append(data_test,"Y")
        else:
            count+=1
            print("N")
            data_test=np.append(data_test,"N")
            
            
        if(count==2):
            print("已達錯誤次數13")
            count=0
            firebase_account = firebase.FirebaseApplication('https://wellsitting-ef0e5.firebaseio.com/', None)
            new=data_test.tolist()#把nparray轉為列表(為JSON serializable)
            returnToDatabase=firebase_account.put('account/'+name,'status',new)
            #put:改寫或寫入 post:新增資料
            
