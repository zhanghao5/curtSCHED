#ifndef TIMING_HPP
#define TIMING_HPP
#ifdef WIN32
#include "gettimeofday.h"
#else
#include <sys/time.h>
#include <sys/stat.h>
#endif

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <map>
#include <cmath>

#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#if !defined(ARRAY_SIZE)
    #define ARRAY_SIZE(x) (sizeof((x)) / sizeof((x)[0]))
#endif

#define WSIZE 50
#define ERROR_BOUND 0
using namespace std;

struct timespec nextRelease;  // initialized in dispatch. TODO: fix, init here?
struct timespec rtPeriod;  // initialized in dispatch. TODO: fix, init here?

struct timespec *getRtPeriodPtr(void) {
  // 1 sec period
  rtPeriod.tv_nsec = 0;
  rtPeriod.tv_sec = 1;
  return &rtPeriod;
}

struct timespec *getNextReleasePtr(void) {
  return &nextRelease;
}

int waitForNextRelease(void) {
  struct timespec remain;
  clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &nextRelease, &remain);
}

// to the number of parameters
size_t tensorSize(const cudnnTensorDescriptor_t tensorDesc) {
  size_t size;
  cudnnGetTensorSizeInBytes(tensorDesc, &size);
  return size/4;
} 

size_t filterSize(const cudnnFilterDescriptor_t filterDesc){
	size_t size;
	cudnnGetFilterSizeInBytes(filterDesc, &size);
	return size/4;
}

// src1 - src2
void minus_clock(timespec *dst, timespec *src1, timespec *src2){
	if (src2->tv_nsec > src1->tv_nsec) {
      	dst->tv_sec = src1->tv_sec - src2->tv_sec - 1;
      	dst->tv_nsec = src1->tv_nsec - src2->tv_nsec + 1000000000; 
    }
    else {
    	dst->tv_sec = src1->tv_sec - src2->tv_sec;
        dst->tv_nsec = src1->tv_nsec - src2->tv_nsec; 
    }
}

void plus_clock(timespec *dst, timespec *src1, timespec *src2){
	if(src1->tv_nsec + src2->tv_nsec > 1000000000){
		dst->tv_sec = src1->tv_sec + src2->tv_sec + 1;
		dst->tv_nsec = -1000000000 + src1->tv_nsec + src2->tv_nsec;
	}
	else{
		dst->tv_sec = src1->tv_sec + src2->tv_sec;
		dst->tv_nsec = src1->tv_nsec + src2->tv_nsec;
	}
}



class sampling_event{
public:
	deque<long> samples;         // sampling queue with fixed szie
	string func;               // monitored kernel function
	long mean;					
	long variance;
	unsigned grid;               // gpu grid dimension
	unsigned block;				 // gpu block dimension
	unsigned qsize;				 // sample queue size or window size
	unsigned cnt;				 // total number of sample > window size
	unsigned max;
	unsigned min;
	unsigned total;
	map<unsigned, unsigned> hist;      // key is sample time /10, 10 interval, value is the counter
	
	sampling_event(string f, unsigned g, unsigned b, unsigned q) {
		mean = 0;
		variance = 0;
		func = f;
		grid = g;
		block = b;
		qsize = q;
		max = 0;
		min = UINT_MAX;
		total = 0;
			
	}

	void add_sample(long e_time){
		// update histogram info
		total++;
		hist[unsigned(e_time/10)]++;

		update_max_min(e_time);
		if( samples.size() ==  qsize){
			long oldest = samples.front();
			samples.pop_front();
			samples.push_back(e_time);
			update_rolling_mv(oldest, e_time);
		}
		else{
			samples.push_back(e_time);
			if(samples.size() == qsize){
				cal_mean();
				cal_variance();
				cnt = qsize;
			}
		}
	}

	void load_mv(long m, long v){
		if(m < 0) {
#ifdef DEBUG
			printf("load_mv --- negtive mean : %ld\n", m);
#endif			
			mean = -m;

		}
		else mean = m;
		if(v < 0) variance = -v;
		else variance = v;
	}

	void load_hist(string a, string b){
		total = stol(a, nullptr, 0);
		size_t pos = 0;
		string token;
		while ((pos = b.find(",")) != std::string::npos) {
    		token = b.substr(0, pos);
			size_t pos0 = b.find(" ");
			unsigned k = stoul(token.substr(0, pos0), nullptr, 0);
			unsigned v = stoul(token.substr(pos0+1), nullptr, 0);
			hist[k] = v;
    		b.erase(0, pos + 1);   // 1 deliminiter length
		}


	}

	void cal_mean(){
		if(samples.empty()){
			mean = 0;
		}
		else{
			mean = accumulate(samples.begin(), samples.end(), 0) / samples.size();

		}
	}
	void cal_variance(){
		// calculate mean first
		for(int i = 0; i < samples.size(); i++){
			variance += (samples[i] - mean) * (samples[i] - mean);
			variance /= samples.size();
		}
#ifdef DEBUG
		   cout<< "cal variance : "<<variance << endl;
#endif

	}
	// update rolling mean and variance when shift window
	void update_rolling_mv(long oldest, long latest){
		long old_mean = mean;
		long old_var = variance;
		mean = old_mean + (latest - oldest)/qsize;
#ifdef DEBUG
		cout << "last first old_mean mean old_val\t" << latest <<"\t" << oldest<<"\t" << old_mean<<"\t"  << mean<<"\t" << old_var << endl;	
#endif
		// method 1
		variance = old_var + (latest - oldest) * (latest + oldest - mean - old_mean);
		// method 2
		// long sx2 = pow(old_mean * qsize, 2) + pow(latest,2) - pow(oldest, 2);
		// variance = sx2/qsize - pow(mean, 2);
		// method 3
		// cnt += 1;
		// long delta = latest - mean;
		// mean += delta / cnt;
		// long m2 = delta * delta;
		// variance = m2 / cnt;
		// long var_2 = m2/(cnt - 1);

#ifdef DEBUG
		cout<< func <<"update rolling mean and variance : " << mean << "\t" << variance << endl;		
#endif
	}

	void update_max_min(long et){
		if(et > max){
			max = et;
		}
		if(et < min){
			min = et;
		}
	}

	long get_mean(){
		return mean;
	}
	long get_var(){
		return variance;
	}

};

typedef pair<unsigned ,unsigned> pair_m;

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};
typedef unordered_map<pair_m, sampling_event*, pair_hash> e_table;
typedef unordered_map<string, e_table> f_table;

class timing_info{
public:
	// func search first
	f_table table;
	unsigned int f_count;   // count total number of cuLaunchkernel function
	unsigned int s_cnt;
	

	timing_info(){
		f_count = 0;
		s_cnt = 0;
	};
	~timing_info(){};
	void record(string f, unsigned g, unsigned b, long time){
		if(table.count(f)){
			// in-table
			if(table[f].count(make_pair(g,b))){
				table[f][make_pair(g,b)]->add_sample(time);
			}
			else{
				sampling_event *se = new sampling_event(f, g ,b, WSIZE);
				table[f].insert(make_pair(make_pair(g,b), se));
				table[f][make_pair(g,b)]->add_sample(time);
				s_cnt++;
#ifdef DEBUG
				printf("current s # : %u\n", s_cnt);
#endif

			}
			
		}
		else{
			sampling_event *se = new sampling_event(f, g ,b, WSIZE);
			table[f].insert(make_pair(make_pair(g,b), se));
			// table[make_tuple(f,g,b)] = new sampling_event(f, g, b, WSIZE);
			// table[make_tuple(f,g,b)].add_sample(time);
			table[f][make_pair(g,b)]->add_sample(time);
			// f_count++;
		    // printf("current f # : %u\n", f_count);
		}

	}

	void save_table(char* filename){
		ofstream out_file(filename);
		if(out_file.is_open()){
			f_table::iterator it;
			for(it = table.begin(); it != table.end(); it++){
				out_file<< it->first
						<< ":"
						<< it->second.size()
						<< "--";
				e_table::iterator itl;
				for(itl = it->second.begin(); itl != it->second.end(); itl++){
					out_file<< itl->first.first
							   << " "
							   << itl->first.second
							   << " "
							   << itl->second->mean
							   << " "
							   << itl->second->max
// add min and max
							   // << "  "
							   // << itl->second->max
							   // << " "
							   // << itl->second->min
// end of adding							   
							   << "\t";
				}
				out_file<<endl;		
			}
			out_file.close();
		}
		else cout<< "unable to open write out file!\n" ;
	}

	void save_histogram(char* filename){
		ofstream out_file(filename);
		if(out_file.is_open()){
			f_table::iterator it;
			for(it = table.begin(); it != table.end(); it++){
				e_table::iterator itl;
				for(itl = it->second.begin(); itl != it->second.end(); itl++){
					out_file<< it->first <<":"
                            << itl->first.first << " " << itl->first.second << ":"
                            << itl->second->total << ":";
                            for (map<unsigned,unsigned>::iterator itr = itl->second->hist.begin(); itr != itl->second->hist.end(); ++itr)
    								out_file << itr->first << " " << itr->second << ",";
    						out_file<<":"<<endl;
				}
			}
			out_file.close();
		}
		else cout<< "failed to open write out histogram!\n" ;
	}

	void load_entry(string id, unsigned g, unsigned b, long m, long v){
		sampling_event *se = new sampling_event(id, g ,b, WSIZE);
		table[id].insert(make_pair(make_pair(g,b), se));
			// table[make_tuple(f,g,b)] = new sampling_event(f, g, b, WSIZE);
			// table[make_tuple(f,g,b)].add_sample(time);
		table[id][make_pair(g,b)]->load_mv(m,v);
	}

	void load_entries(string id, unsigned loopn, string &values){
		size_t pos0 = 0;

#ifdef DEBUG
		cout<< "load entries id : " << id << endl;
#endif
		for(int i = 0; i < loopn; i++){
			pos0 = values.find("\t");
			string value = values.substr(0, values.find("\t"));
			vector<long> uv;
			size_t pos = 0;
			string token;
			while ((pos = value.find(" ")) != std::string::npos) {
    			token = value.substr(0, pos);
#ifdef DEBUG
    			std::cout <<"token value : " << token << endl;
#endif    		
				long tmp = stol(token, nullptr, 0);
				uv.push_back(tmp);
    			value.erase(0, pos + 1);
			}
			long last = stol(value, nullptr, 0);
#ifdef DEBUG	
			std::cout <<"last value : " << last << endl;
#endif			
			uv.push_back(last);
			load_entry(id, uv[0], uv[1], uv[2], uv[3]);
			values.erase(0, pos0 + 1);
		}
	}

	void load_table(char* filename){
		ifstream in_file(filename);
		string line;
		if(in_file.is_open()){
			while( getline(in_file, line)){
				string first = line.substr(0, line.find("--"));
				string fid = first.substr(0, first.find(":"));
				string n = first.substr(first.find(":") + 1, first.length() - 1);
				unsigned long loopn = stoul(n, nullptr, 0);
#ifdef DEBUG
				cout << "id : " << fid << "  loop size : " << loopn << endl;
#endif			
				string values = line.substr(line.find("--")+2, line.length()-1);
				load_entries(fid, loopn, values);

			}


		}
		else cout<< "unable to open input file!\n";

	}

	long get_mean_time(string id, unsigned g, unsigned b){
		if(table.count(id)){
			// in-table
			if(table[id].count(make_pair(g,b))){
				return table[id][make_pair(g,b)]->get_mean();
			}
			else{
				printf("cannot find the entry in the sub table\n");
				return 1000;
			}
		}
		else{
			printf("cannot find the entry in the main table\n");
			return 1000;
		}
	}

	void load_hist_entry(vector<string> v){
		unsigned g,b;
		size_t pos = 0;
		pos = v[1].find(" ");
		g = stol(v[1].substr(0, pos), nullptr, 0);
		b = stol(v[1].substr(pos+1), nullptr, 0 );
		sampling_event *se = new sampling_event(v[0], g ,b, WSIZE);
		table[v[0]].insert(make_pair(make_pair(g,b), se));
		table[v[0]][make_pair(g,b)]->load_hist(v[2],v[3]);
	}

	void load_hist_table(char* filename){
		ifstream in_file(filename);
		string line;
		if(in_file.is_open()){
			while( getline(in_file, line)){
				size_t pos = 0;
				string token;
				vector<string> load_vec;
				while ((pos = line.find(":")) != std::string::npos) {
    				token = line.substr(0, pos);
#ifdef DEBUG
    				std::cout <<"token value : " << token << endl;
#endif    		
                	load_vec.push_back(token);
    				line.erase(0, pos + 1);   // 1 deliminiter length
				}
				load_hist_entry(load_vec);
			}
		}
		else cout<< "unable to open input file!\n";

	}

	long get_score_value(map<unsigned, unsigned>hist, unsigned total,  float score){
		unsigned acc_cnt = 0;
		for(auto const&x : hist){
			acc_cnt += x.second;
			if(float(acc_cnt/total) > score){
				unsigned less_cnt = 0; // less than threshhold cnt in this interval
				for(int i = 0; i < x.second; i++){
					if(float(acc_cnt - x.second + i)/float(total) >= score){
						less_cnt = i;
						break;
					}
				}
				return (unsigned)(float(less_cnt)/float(x.second) *10 ) + x.first * 10;   // 10 is the interval of histogram
			}
		}

	}

	// call after load_hist_table, otherwise, no value update
	void confidence_knob(float score){
		for(f_table::iterator it = table.begin(); it != table.end(); it++){
				for(e_table::iterator itl = it->second.begin(); itl != it->second.end(); itl++){
					itl->second->mean = get_score_value(itl->second->hist, itl->second->total,  score);
				}
		}

	}


};

#endif