#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_DIM 2
#define OUTPUT_DIM 2
#define HIDDEN_DIM 16
#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define N_SAMPLES 1000

// 矩阵结构体
typedef struct{
    int rows;
    int cols;
    double** data;
}Matrix;

// 1.创建矩阵
Matrix* create_matrix(int rows,int cols){
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double**)malloc(rows*sizeof(double*));
    for(int i = 0;i<rows;i++){
        mat->data[i] = (double*)calloc(cols,sizeof(double));
    }
    return mat;
}

// 2.释放矩阵内存
void free_matrix(Matrix* mat){
    for(int i=0;i<mat->rows;i++){
        free(mat->data[i]);
    }
    free(mat->data);
    free(mat);
}

// 3.矩阵乘法
Matrix* matrix_multiply(Matrix* A,Matrix* B){
    Matrix* C = create_matrix(A->rows,B->cols);
    for(int i=0;i<A->rows;i++){
        for(int j=0;j<B->cols;j++){
            C->data[i][j] = 0;  // 初始化累加值
            for(int k=0;k<A->cols;k++){
                C->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return C;
}

// 4.生成二分类数据
void generate_data(Matrix* X,int* y){
    srand(42);
    for(int i=0;i<N_SAMPLES;i++){
        double t = (rand()%1000)/1000.0*2*M_PI;
        double x1 = cos(t) + (rand()%200 - 100)/1000.0;
        double x2 = sin(t) + (rand()%200 - 100)/1000.0;
        if(i%2==0){
            x1 = 1 - cos(t) + (rand()%200 - 100)/1000.0;
            x2 = 0.5 - sin(t) + (rand()%200 - 100)/1000.0;
        }
        X->data[i][0] = x1;
        X->data[i][1] = x2;
        y[i] = (i%2==0)?1:0;
    }
}

// 5.神经网络结构体
typedef struct{
    Matrix* W1;
    Matrix* b1;
    Matrix* W2;
    Matrix* b2;
}NeuralNetwork;

// 6.初始化神经网络参数
NeuralNetwork* init_network(){
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->W1 = create_matrix(INPUT_DIM,HIDDEN_DIM);
    nn->b1 = create_matrix(1,HIDDEN_DIM);
    nn->W2 = create_matrix(HIDDEN_DIM,OUTPUT_DIM);
    nn->b2 = create_matrix(1,OUTPUT_DIM);
    int i,j;
    for(i=0;i<INPUT_DIM;i++){
        for(j=0;j<HIDDEN_DIM;j++){
            nn->W1->data[i][j] = (rand()%2000-1000)/100000.0;
        }
    }
    for(i=0;i<HIDDEN_DIM;i++){
        for(j=0;j<OUTPUT_DIM;j++){
            nn->W2->data[i][j] = (rand()%2000-1000)/100000.0;
        }
    }
    return nn;
}

// 7.前向传播
void forward(NeuralNetwork* nn,Matrix* X,Matrix* probs,Matrix* A1){
    Matrix* Z1 = matrix_multiply(X,nn->W1);
    for(int i=0;i<Z1->rows;i++){
        for(int j=0;j<Z1->cols;j++){
            Z1->data[i][j] += nn->b1->data[0][j];
            A1->data[i][j] = (Z1->data[i][j]>0)?Z1->data[i][j]:0;
        }
    }
    Matrix* Z2 = matrix_multiply(A1,nn->W2);
    for(int i=0;i<Z2->rows;i++){
        for(int j=0;j<Z2->cols;j++){
            Z2->data[i][j] += nn->b2->data[0][j];
        }
    }
    // 计算Softmax
    for(int i=0;i<Z2->rows;i++){
        double max_val = Z2->data[i][0];
        for(int j=1;j<Z2->cols;j++){
            if(Z2->data[i][j]>max_val){
                max_val = Z2->data[i][j];
            }
        }
        double sum_exp = 0;
        for(int j=0;j<Z2->cols;j++){
            sum_exp += exp(Z2->data[i][j]-max_val);
        }
        for(int j=0;j<Z2->cols;j++){
            probs->data[i][j] = exp(Z2->data[i][j]-max_val)/sum_exp;
        }
    }
    free_matrix(Z1);
    free_matrix(Z2);
}

// 8.计算交叉熵损失
double compute_loss(Matrix* probs,int* y){
    double total_loss = 0;
    for(int i=0;i<probs->rows;i++){
        int label = y[i];
        total_loss -= log(probs->data[i][label] + 1e-10);  // 提高数值稳定性
    }
    return total_loss/probs->rows;
}

// 9.计算准确率
double compute_accuracy(Matrix* probs,int* y){
    int correct = 0;
    for(int i=0;i<probs->rows;i++){
        int pred = (probs->data[i][0] > probs->data[i][1]) ? 0 : 1;
        if(pred == y[i]){
            correct++;
        }
    }
    return (double)correct / probs->rows;
}

// 10.反向传播
void backward(NeuralNetwork* nn, Matrix* X, Matrix* A1, Matrix* probs, int* y){
    int n_samples = X->rows;
    Matrix* dZ2 = create_matrix(n_samples,OUTPUT_DIM);
    for(int i=0;i<n_samples;i++){
        for(int j=0;j<OUTPUT_DIM;j++){
            dZ2->data[i][j] = probs->data[i][j];
        }
        dZ2->data[i][y[i]] -= 1;
        for(int j=0;j<OUTPUT_DIM;j++){
            dZ2->data[i][j] /= n_samples;
        }
    }

    // 计算W2和b2的梯度
    Matrix* A1_T = create_matrix(A1->cols,A1->rows);
    for(int i=0;i<A1->rows;i++){
        for(int j=0;j<A1->cols;j++){
            A1_T->data[j][i] = A1->data[i][j];
        }
    }
    Matrix* dW2 = matrix_multiply(A1_T,dZ2);
    Matrix* db2 = create_matrix(1,OUTPUT_DIM);
    for(int j=0;j<OUTPUT_DIM;j++){
        for(int i=0;i<n_samples;i++){
            db2->data[0][j] += dZ2->data[i][j];
        }
    }

    // 隐藏层梯度
    Matrix* W2_T = create_matrix(OUTPUT_DIM,HIDDEN_DIM);
    for(int i=0;i<HIDDEN_DIM;i++){
        for(int j=0;j<OUTPUT_DIM;j++){
            W2_T->data[j][i] = nn->W2->data[i][j];
        }
    }
    Matrix* dA1 = matrix_multiply(dZ2,W2_T);
    Matrix* Z1 = matrix_multiply(X, nn->W1);
    for (int i = 0; i < Z1->rows; i++) {
        for (int j = 0; j < Z1->cols; j++) {
            Z1->data[i][j] += nn->b1->data[0][j];
        }
    }
    Matrix* dZ1 = create_matrix(n_samples,HIDDEN_DIM);
    for(int i=0;i<n_samples;i++){
        for(int j=0;j<HIDDEN_DIM;j++){
            dZ1->data[i][j] = dA1->data[i][j] * (Z1->data[i][j] > 0 ? 1 : 0);
        }
    }

    // 计算W1和b1的梯度
    Matrix* X_T = create_matrix(INPUT_DIM,n_samples);
    for(int i=0;i<n_samples;i++){
        for(int j=0;j<INPUT_DIM;j++){
            X_T->data[j][i] = X->data[i][j];
        }
    }
    Matrix* dW1 = matrix_multiply(X_T,dZ1);
    Matrix* db1 = create_matrix(1,HIDDEN_DIM);
    for (int j = 0; j < HIDDEN_DIM; j++) {
        for (int i = 0; i < n_samples; i++) {
            db1->data[0][j] += dZ1->data[i][j];
        }
    }

    // 更新参数
    for(int i=0;i<INPUT_DIM;i++){
        for(int j=0;j<HIDDEN_DIM;j++){
            nn->W1->data[i][j] -= LEARNING_RATE * dW1->data[i][j];
        }
    }
    for(int j=0;j<HIDDEN_DIM;j++){
        nn->b1->data[0][j] -= LEARNING_RATE * db1->data[0][j];
    }
    for(int i=0;i<HIDDEN_DIM;i++){
        for(int j=0;j<OUTPUT_DIM;j++){
            nn->W2->data[i][j] -= LEARNING_RATE * dW2->data[i][j];
        }
    }
    for(int j=0;j<OUTPUT_DIM;j++){
        nn->b2->data[0][j] -= LEARNING_RATE * db2->data[0][j];
    }
    
    // 释放临时矩阵
    free_matrix(dZ2);
    free_matrix(A1_T);
    free_matrix(dW2);
    free_matrix(db2);
    free_matrix(W2_T);
    free_matrix(dA1);
    free_matrix(Z1);
    free_matrix(dZ1);
    free_matrix(X_T);
    free_matrix(dW1);
    free_matrix(db1);
}

// 11.保存矩阵到文件
void save_matrix(Matrix* mat,const char* filename){
    FILE* file = fopen(filename,"w");
    if(file == NULL){
        printf("无法打开文件 %s\n",filename);
        return;
    }
    fprintf(file,"%d %d\n",mat->rows,mat->cols);
    for(int i = 0;i< mat->rows;i++){
        for(int j=0;j < mat->cols;j++){
            fprintf(file,"%.6f ",mat->data[i][j]);
        }
        fprintf(file,"\n");
    }
    fclose(file);
}

// 12.保存训练日志
void save_train_log(double* losses,double* accs){
    FILE* file = fopen("train_log.txt","w");
    for(int epoch = 0;epoch < EPOCHS;epoch++){
        fprintf(file,"%d %.6f %.6f\n",epoch,losses[epoch],accs[epoch]);
    }
    fclose(file);
}

// 13.主函数：训练网络
int main(){
    Matrix* X = create_matrix(N_SAMPLES,INPUT_DIM);
    int* y = (int*)malloc(N_SAMPLES*sizeof(int));
    generate_data(X,y);
    
    NeuralNetwork* nn = init_network();
    Matrix* probs = create_matrix(N_SAMPLES,OUTPUT_DIM);
    Matrix* A1 = create_matrix(N_SAMPLES,HIDDEN_DIM);
    double* losses = (double*)malloc(EPOCHS* sizeof(double));
    double* accs = (double*)malloc(EPOCHS * sizeof(double));
    
    clock_t start_time = clock();
    for(int epoch = 0;epoch < EPOCHS;epoch++){
        forward(nn,X,probs,A1);
        losses[epoch] = compute_loss(probs,y);
        accs[epoch] = compute_accuracy(probs,y);
        backward(nn,X,A1,probs,y);
        
        if(epoch % 100 == 0){
            printf("Epoch %d | Loss: %.4f | Accuracy: %.4f\n",epoch,losses[epoch],accs[epoch]);
        }
    }
    clock_t end_time = clock();
    printf("训练完成！总耗时：%.2f秒\n",(double)(end_time - start_time)/1000000.0);
    
    save_matrix(nn->W1,"W1.txt");
    save_matrix(nn->W2,"W2.txt");
    save_train_log(losses,accs);
    
    // 释放所有动态分配的内存
    free_matrix(X);
    free_matrix(probs);
    free_matrix(A1);
    free_matrix(nn->W1);
    free_matrix(nn->b1);
    free_matrix(nn->W2);
    free_matrix(nn->b2);
    free(nn);
    free(y);
    free(losses);
    free(accs);
    
    return 0;
}
