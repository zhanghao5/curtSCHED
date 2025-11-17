#include "model.hpp"

/* this is for unit test model class only 
   
   will not reflect in the dispatch.so
*/

int main(){
	Model m(2); // two features

	m.init_weights(2,1);
	int max_iteration = 1000;
    float learning_rate = 0.1;
    for(int i = 0; i < 20; i++){
    	m.X.push_back({i, i+1});
    	m.Y.push_back(i*100);
    }
    
    m.train(max_iteration, learning_rate);
    std::cout<<"weights: " << m.weights[0] << '\t' << m.weights[1] << std::endl;

}