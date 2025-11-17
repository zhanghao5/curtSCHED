#ifndef MYTIME_HPP
#define MYTIME_HPP

#include<vector>
#include<iostream>

using namespace std;

class Model{
public:
	vector<vector<int>> X;			// input X, size info
	vector<float> Y;					// output Y, execution time
	int nfeatures;					// # of features or n variate
	std::vector<float> weights;   // can be multivariate LR
	int MAX;					  // max weigh value
    
	Model();
	Model(int n){
		nfeatures = n;
	}
	void init_weights(int nweights, int random){
		MAX = 1000;
		for(int i = 0; i < nweights; i++){
			if(random){
				weights.push_back(rand() % MAX);
			}
			else{
				weights.push_back(0.0);
			}
		}
	}

	void update_weights(float* y_p, float learning_rate){
		float multiplier = learning_rate / X.size();
		for(int i = 0; i < nfeatures; i++){
        	float sum = (sum_residual(y_p,i));
        	printf("update sum = %f\n",sum);
        	weights[i] = weights[i] - multiplier*sum; 
		}
	}

	// ci current feature index
    float sum_residual(float *y_pred, int ci){
    	float total = 0;
    	float residual;
    	for(int i = 0 ; i < X.size(); i++){
        	residual = (y_pred[i] - Y[i]);
        	total = total + residual*X[i][ci];
    	}
    	return total;
    }

    float residual_sum_of_square(float *y_pred, vector<float> &y_true, int length){
        float total = 0;
        float residual;
        for(int i = 0 ; i < length; i++){
            residual = (y_true[i] - y_pred[i]);
            total = total + (residual*residual);
        }
        return total;
    }    

    float mean_squared_error(float *y_pred, vector<float> &y_true, int length){
        return residual_sum_of_square(y_pred,y_true,length)/length;
    }

    // Train LR
    void train(int max_iteration, float learning_rate){

        // Mallocating some space for prediction
        float *y_pred = (float *) std::malloc(sizeof(float)*X.size());
			while(max_iteration > 0){
                fit(y_pred);
                update_weights(y_pred, learning_rate);
				float mse = mean_squared_error(y_pred,Y,X.size());
                if(max_iteration % 100 == 0){
                    //print_weights();
                    std::cout << "Iteration left: " << max_iteration << "; MSE = " << mse << "\n";
                }
                max_iteration--;
            }
        free(y_pred);
    }
        
    float predict(vector<int> x){
        float prediction = 0;
        for(int i = 0; i < nfeatures; i++){
            prediction = prediction + weights[i]*x[i];
        }
        return prediction;
    }


    // fit a line given some x and weights
    void fit(float *y_pred){
        for(int i = 0; i < X.size(); i++){
            y_pred[i] = predict(X[i]);
        }
    }

};

#endif