#include "utils.hpp"

static bool isSubstring(const char *substring, const char *string) {
    int subLen = strlen(substring);
    int strLen = strlen(string);

    for (int i = 0; i <= strLen - subLen; i++) {
        int j;
        for (j = 0; j < subLen; j++) {
            if (string[i + j] != substring[j]) {
                break;
            }
        }
        if (j == subLen) {
            return true; // Match found
        }
    }

    return false; // Match not found
}


int findInStack ( char *fname) {
  int ret;
  unw_cursor_t cursor;
  unw_context_t context;

  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  int n=0;
  while ( unw_step(&cursor) ) {
    unw_word_t ip, sp, off;

    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    unw_get_reg(&cursor, UNW_REG_SP, &sp);

    char symbol[256] = {"<unknown>"};
    char *name = symbol;

    if ( !unw_get_proc_name(&cursor, symbol, sizeof(symbol), &off) ) {
      int status;
      if ( (name = abi::__cxa_demangle(symbol, NULL, NULL, &status)) == 0 )
        name = symbol;
    }
#ifdef DEBUG
    printf("#%-2d 0x%016" PRIxPTR " sp=0x%016" PRIxPTR " %s + 0x%" PRIxPTR "\n",
        ++n,
        static_cast<uintptr_t>(ip),
        static_cast<uintptr_t>(sp),
        name,
        static_cast<uintptr_t>(off));
#endif
    // now the name is add .localalias
    ret = (strcmp(fname, name) == 0);  /* exact func name match */
    if(isSubstring(fname, name))     /* sub string match */
    {
        ret = 1;
    }
    else ret = 0;

    if ( name != symbol )
      free(name);

    if (ret)
      return ret;
  }
  return ret;
}

/* load the from the configuration file rather than hard code */
/* 1) Mode tell if this is a profiling/training timing model mode (0) or a online schedule mode (1)
 * 2) guaranteed testing speedd in nano sec per image, 33333333 for 30 FPS
 * 3) realtime calls per image for different GPU archs
 * 4) confidence score for prediction execution time from histogram 
 *  */ 
void load_config(const char* filename, cu_config* config) {
   ifstream in_file(filename);
    string line;
    if(in_file.is_open()){
        while( getline(in_file, line)){
            string key = line.substr(0, line.find("="));
            string value = line.substr(line.find("=") + 1);        
            if(strcmp(key.c_str(),"mode") == 0 ) {
                config->mode = stol(value, nullptr, 0); 
            }
            else if(strcmp(key.c_str(),"period") == 0 ){
                config->period.tv_sec = 0;
                config->period.tv_nsec = stoul(value);
            }
            else if(strcmp(key.c_str(),"realtimecalls") == 0 ){
                config->n_calls = stol(value, nullptr, 0);
            }
            else if(strcmp(key.c_str(), "confidence") == 0 ){
                config->confidence = stof(value);
            } 
        }
    }
    else cout<< "unable to open input file!\n";
}



