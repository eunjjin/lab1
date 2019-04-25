Lab 05-3: Logistic Regression/Classification 를 TensorFlow 로 구현하기
x_train = [[1., 2.],
[2., 3.],
[3., 1.],
[4., 3.],
[5., 3.],
[6., 2.]]
y_train = [[0.],
[0.],
[0.],
[1.],
[1.],
[1.]]

x_test = [[5.,2.]]
y_test = [[1.]]

→ 학습을 위한 x 데이터 y 데이터
import tensoflow.contrib.eager as tfe     
   #tensorflow에서 eager모드 실행을 위한 라이브러리 실행  
tf.enable_eager_execution()        
 #eager 모드 실행을 위해 eage excution을 먼저 선언한다  dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))     
#가져올 데이터 즉 tf데이타를 통해서 우리가 원하는 x값과 y값을 실제  
x의 길이만큼 batch학습을 하겠다는것을 토대로 데이터 값을 가져온다. 
W = tf.Variable(tf.zeros([2, 1]), name='weight')
#w는 2행1열이고 이름은 'weight'로 지정한다. 
 b = tf.Variable(tf.zeros([1]), name='bias')  
  #b값을 지정하고 이름은 'bias'로 한다. 

 def logistic_regression(features):
    hypothesis  = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))              
 # 리니어한 값을 시그모이드함수를 통해 가설을 설정할수있다
    return hypothesis

def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))
    return cost                                                                     # cost값을 구하기 위한 lable과 hypothesis 값!


def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels)        
 # 가설과 실제 값을 비교한 loss 값을 구하고
    return tape.gradient(loss_value, [W,b])
 # gradient로 값을 계속 변화시킴
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)                  
 #learning_rate한 값으로 실제 우리가 이동할 러닝레이트를 통한 값으로 optimizer를 선언한다. 


#지금까지 선언한 함수들을 실제 학습을 위해 호출
for step in range(EPOCHS): 
        
    for features, labels  in tfe.Iterator(dataset):                                 
# dataset을 가져와서 데이터를 토대로 Iterator를 돌려서 x,y값을 넣어가며 모델이 만들어짐
        grads = grad(logistic_regression(features), features, labels)   
 #x값과 y값이 나오게 된 것을 실제 가설을 집어넣어서 학습을 위한 grads값이 나오게된다. 
 optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))      
 #optimizer를 통해 계속 minimize하는 것을 구한다. 이과정을 통해 w,b가 업데이트 되면서 업데이트하며 내려가면서 최적의 값을 나타낼수 있게된다. 
if step % 100 == 0:                   
 #100번마다 Iter와 Loss값이 줄어드는 것을 출력한다. 
print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))  
 
#정확하게 그어지는 모델이 맞는지를 새로운 데이터를 넣어 확인            
def accuracy_fn(hypothesis, labels):              
#accuracy_fn: hypothesis와 labels를비교하기 위한 함수선언 
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)        
#x값을 넣었을때의 hypothesis는 logistic fn을 통해 나온값임 
#모델이 0과1을 결정하기 위한 구간을 hypothesis가 나온값이 시그모이드의 1과 0의 사이로 나온것을 0.5로 통해 우리가 예측된 값(predicted)이 나오게됨 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))  
 #실제 값과 예측되어 나온값을 비교해서 이값이  
실제로 맞는지 안맞는지 accuracy를 출력하게 되는것임
 return accuracy test_acc = accuracy_fn(logistic_regression(x_test),y_test)    
#test_acc자체를 x_test와 y_test를 넣어서 출력할수있게됨 
     
