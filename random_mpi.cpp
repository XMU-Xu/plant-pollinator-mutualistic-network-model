 //paper is Predicting tipping points in mutualistic networks through dimension reduction
// by Xu Fei 2022.5.31
#include <fstream> 
#include <iostream>
#include <typeinfo>
#include <math.h>
#include <string>
//#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include<sys/file.h>
#include"mpi.h"

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/* Period parameters */  
#define NNNN 624
#define MMMM 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

static unsigned long state[NNNN]; /* the array for the state vector  */
static int left = 1;
static int initf = 0;
static unsigned long *next;

/* initializes state[NNNN] with a seed */
void init_genrand(unsigned long s)
{
    int j;
    state[0]= s & 0xffffffffUL;
    for (j=1; j<NNNN; j++) {
        state[j] = (1812433253UL * (state[j-1] ^ (state[j-1] >> 30)) + j); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array state[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        state[j] &= 0xffffffffUL;  /* for >32 bit machines */
    }
    left = 1; initf = 1;
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array(unsigned long init_key[], unsigned long key_length)
{
    int i, j, k;
    init_genrand(19650218UL);
    i=1; j=0;
    k = (NNNN>key_length ? NNNN : key_length);
    for (; k; k--) {
        state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1664525UL)) + init_key[j] + j; /* non linear */
        state[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=NNNN) { state[0] = state[NNNN-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=NNNN-1; k; k--) {
        state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        state[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=NNNN) { state[0] = state[NNNN-1]; i=1; }
    }

    state[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
    left = 1; initf = 1;
}

static void next_state(void)
{
    unsigned long *p=state;
    int j;

    /* if init_genrand() has not been called, */
    /* a default initial seed is used         */
    if (initf==0) init_genrand(5489UL);

    left = NNNN;
    next = state;
    
    for (j=NNNN-MMMM+1; --j; p++) 
        *p = p[MMMM] ^ TWIST(p[0], p[1]);

    for (j=MMMM; --j; p++) 
        *p = p[MMMM-NNNN] ^ TWIST(p[0], p[1]);

    *p = p[MMMM-NNNN] ^ TWIST(p[0], state[0]);
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/* generates a random number on [0,0x7fffffff]-interval */
long genrand_int31(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (long)(y>>1);
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double genrand_real3(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return ((double)y + 0.5) * (1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(void) 
{ 
    unsigned long a=genrand_int32()>>5, b=genrand_int32()>>6; 
    return(a*67108864.0+b)*(1.0/9007199254740992.0); 
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#define pi 3.141592653
#define NT 1E6
#define N  201
#define NS 10
#define NA 500
#define SP 10
#define SA 38
#define dt 1e-3
#define NP 1000 

int it, i, ii, j, jj, tt, k, i_plant, i_pollinator, i_kappa;
int n_data, j_n, j_s, j_a, j_ie;
double j_m;

double P;
double A;

std::vector< std::vector< double > > abundance(2, std::vector< double > (N, 0.0) );

double star_kappa, end_kappa;

double dP;
double dA;

double init_P;
double init_A;

std::vector<double> R_Peff(NA);
std::vector<double> R_Aeff(NA);

double sum_PP;
double sum_AA;

std::vector< std::vector< double > > epsilon(SP, std::vector< double > (SA, 0.0) );
std::vector< std::vector< double > > new_epsilon(SP, std::vector< double > (SA) );

double alpha_P, alpha_A;

//bif variable
std::vector<double> kappa(N);
//std::vector<double> alpha_average(N);

double gc_beta;
double mu;
double tau;
double h;
double gamma_0;
double K_n;

std::vector<double> K_P(SP);
std::vector<double> K_A(SA);

double Sum_K_P;
double Sum_K_A;
double Sum_gamma_P;
double Sum_gamma_A;
double Average_gamma_P;
double Average_gamma_A;
double gamma_Plant;
double gamma_Pollinator;

std::vector<double> sigma(NS);

std::vector<double> RN(SA);

double t;

double rand_numa1, rand_numb1;

std::vector< std::vector< double > > kappa_data(NP, std::vector< double > (35, 0.0) );

FILE *size_file;

int main(int argc, char* argv[])
{
	
//=============================================================================================================	
	MPI_Init(&argc, &argv);//MPI库的初始化
	int numprocs, myid;//定义进程总数及进程标识
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);//获得进程数
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);//获得当前进程标识号0,1,2,3,....,numprocs - 1
//============================================================================================================= 
	
	
	alpha_P=0.1; mu=0.0001; tau=0.5; h=0.4; gamma_0=1.0; gc_beta=1.0;  //kappa=0.0;beta=1.0; alpha_A=-0.3;
	
	srand((unsigned)time(NULL));
	unsigned long init[4]={rand(), rand(), rand(), rand()},  length=4;
	//unsigned long init[4]={0x283, 0x175, 0x615, 0x954},length=4;
	init_by_array(init, length);
	
	float paras[NP][2]={0.0};
	std::ifstream infile;
	infile.open("LHS_of_paras.dat");
	for(i=0;i<NP;i++)
	{
		for(j=0;j<2;j++)
		{
			infile >> paras[i][j];
		}
	}
	infile.close();
	
	size_file = fopen("output.csv","a");
	
	for(i_plant= myid;i_plant<NP;i_plant+= numprocs)
	{
		gamma_Pollinator = paras[i_plant][0];
		gamma_Plant= paras[i_plant][1];
		
		std::vector< std::vector< double > > kappa_data(NP, std::vector< double > (35, 0.0) );
		
		printf("========gamma_Plant=%lf=====gamma_Pollinator=%f=============\n",gamma_Plant,gamma_Pollinator);
		
		for(j_n=0;j_n<2;j_n++)
		{
			for(j_ie=0;j_ie<N;j_ie++)
			{
				if(j_n==0)
					kappa[j_ie]=0.005*j_ie;
				else
					kappa[j_ie]=1.0-0.005*j_ie;
					
				if(j_ie==0)
				{
					P = 1.0; 
					A = 1.0; 
				}
				else
				{
					P = init_P; 
					A = init_A; 
				}
				alpha_A = alpha_P;
				for(it=0;it<=NT;it++)
				{
					t=it*dt;
					
					dA = (alpha_A - kappa[j_ie] - A + gamma_Pollinator*P/(1.0+h*gamma_Pollinator*P))*A + mu;
					dP = (alpha_P - P + gamma_Plant*A/(1.0+h*gamma_Plant*A))*P + mu;
					
					A = A + dA*dt;
					P = P + dP*dt;

				}   // loop of time
				
				init_A = A;
				init_P = P;
				
				abundance[j_n][j_ie] = A+P;
			}
		}
		
		star_kappa = 0.0;
		end_kappa = 0.0;
		int num_bif = 0;
		for(j_ie=0;j_ie<N;j_ie++)
		{
			if(num_bif==0 && fabs(abundance[0][j_ie]-abundance[1][N-1-j_ie]) > 0.05)
			{
				num_bif += 1;
				star_kappa = 0.005*j_ie;
			}
			
			if(num_bif==1 && fabs(abundance[0][j_ie]-abundance[1][N-1-j_ie]) < 0.05)
			{
				num_bif = 0;
				end_kappa = 0.005*j_ie;
				
				printf("=====star_kappa=%lf====end_kappa=%lf=====size_bif=%lf=======\n",star_kappa,end_kappa,end_kappa-star_kappa);
				break;
			}
		}
		
		kappa_data[i_plant][0] = gamma_Plant;
		kappa_data[i_plant][1] = gamma_Pollinator;
		kappa_data[i_plant][2] = star_kappa;
		kappa_data[i_plant][3] = end_kappa;
		kappa_data[i_plant][4] = end_kappa-star_kappa;
		
		if(end_kappa-star_kappa > 0.05)
		{
			
//			std::string fname = "gamma_Plant_" + std::to_string(gamma_Plant) + "_gamma_Pollinator_" + std::to_string(gamma_Pollinator) + ".csv";
//			std::ofstream output_file(fname.c_str());
//			
//			char out_char[100];
//			sprintf(out_char,"%.6lf,%.6lf\n",0.0,end_kappa-star_kappa);
//			output_file << out_char;
			
			for(j_s=0;j_s<NS;j_s++)
			{
				sigma[j_s]=0.01*(j_s+1);
				printf("=============sigma[%d]=%lf===================\n",j_s,sigma[j_s]);
				
				for(j_n=0;j_n<2;j_n++)
				{
//						std::string fname = std::to_string(j_s) + "_Peff-Aeff_" + std::to_string(j_n+1) + ".csv";
//						std::ofstream output_file(fname.c_str());
			
					for(j_ie=0;j_ie<N;j_ie++)
					{
						if(j_n==0)
							kappa[j_ie]=0.005*j_ie;
						else
							kappa[j_ie]=1.0-0.005*j_ie;
						
						sum_PP=0.0; sum_AA=0.0;
						
						for(j_a=0;j_a<NA;j_a++)
						{
							
							if(j_ie==0)
							{
								P = 1.0; 
								A = 1.0; 
							}
							else
							{
								P = init_P; 
								A = init_A; 
							}
							
							while(1)
							{
								rand_numa1 = genrand_real1(); rand_numb1 = genrand_real1();
								alpha_A = (alpha_P + (sqrt(-2.0*log(rand_numa1)) * cos(2.0*pi*rand_numb1))*sigma[j_s]);
								if(alpha_A<1 && alpha_A>-1) break;
							}
							
							for(it=0;it<=NT;it++)
							{
								t=it*dt;
								
								dA = (alpha_A - kappa[j_ie] - A + gamma_Pollinator*P/(1.0+h*gamma_Pollinator*P))*A + mu;
								dP = (alpha_P - P + gamma_Plant*A/(1.0+h*gamma_Plant*A))*P + mu;
								
								A = A + dA*dt;
								P = P + dP*dt;
			
							}   // loop of time
							
							init_A = A;
							init_P = P;
							
							//printf("==alpha_A=%lf====kappa=%.10lf=======P=%.10lf====A=%.10lf====\n",alpha_A,kappa[j_ie],P,A);
							
							R_Peff[j_a] = P;
							R_Aeff[j_a] = A;
										
						}
						
						for(j_a=0;j_a<NA;j_a++)
						{
							sum_PP += R_Peff[j_a];
							sum_AA += R_Aeff[j_a];
						}
						
						if(j_ie%50==0)
							printf("==alpha_A=%lf====kappa=%.10lf=======sum_PP=%.10lf====sum_AA=%.10lf====\n",alpha_A,kappa[j_ie],sum_PP/NA,sum_AA/NA);
						
						abundance[j_n][j_ie] = sum_PP/NA + sum_AA/NA;
						
//							char out_char[100];
//							if(j_ie%2==0)
//							{
//								sprintf(out_char, "%.10lf,%.10lf,%.10lf\n",kappa[j_ie],sum_PP/NA,sum_AA/NA);
//								output_file << out_char;
//							}
						
					}   //loop of N
//						output_file.close();
				}
				
				star_kappa = 0.0;
				end_kappa = 0.0;
				int num_bif = 0;
				for(j_ie=0;j_ie<N;j_ie++)
				{
					if(num_bif==0 && fabs(abundance[0][j_ie]-abundance[1][N-1-j_ie]) > 0.01)
					{
						num_bif += 1;
						star_kappa = 0.005*j_ie;
					}
					
					if(num_bif==1 && fabs(abundance[0][j_ie]-abundance[1][N-1-j_ie]) < 0.01)
					{
						num_bif = 0;
						end_kappa = 0.005*j_ie;
						
						printf("=====star_kappa=%lf====end_kappa=%lf=====size_bif=%lf=======\n",star_kappa,end_kappa,end_kappa-star_kappa);
						break;
					}
				}
				
				if(end_kappa-star_kappa > 0.05)
				{
					kappa_data[i_plant][5 + j_s*3] = star_kappa;
					kappa_data[i_plant][6 + j_s*3] = end_kappa;
					kappa_data[i_plant][7 + j_s*3] = end_kappa-star_kappa;
				}
				else
				{
					break;
				}
			} //loop of NS
			//output_file.close();
		}
		if(0 == flock(fileno(size_file),LOCK_EX))
		{
			fseek(size_file, 0, SEEK_END);
			for(i_kappa=0;i_kappa<35;i_kappa++)
			{
				if(i_kappa<34)
					fprintf(size_file,"%.6lf,",kappa_data[i_plant][i_kappa]);
				else
					fprintf(size_file,"%.6lf\n",kappa_data[i_plant][i_kappa]);
			}
			
			flock(fileno(size_file),LOCK_UN); 
		}
	}
	fclose(size_file);
	
	MPI_Finalize();
	return 0; 
}

