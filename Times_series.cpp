 //paper is Predicting tipping points in mutualistic networks through dimension reduction
// by Xu Fei 2022.5.31
#include <fstream> 
#include <iostream>
#include <typeinfo>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

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
#define NT 2E4
#define N  201
#define NS 16
#define NA 100
#define SP 11
#define SA 38
#define dt 0.01

int it, i, ii, j, jj, tt, k;
int n_data, j_n, j_s, j_a, j_ie;
double j_m;

std::vector<double> P(SP);
std::vector<double> A(SA);

std::vector<double> temp_P(SP);
std::vector<double> temp_A(SA);

std::vector<double> init_P(SP);
std::vector<double> init_A(SA);

double Sum_A, Sum_P;

std::vector<double> R_Peff(NA);
std::vector<double> R_Aeff(NA);

std::vector< std::vector< double > > sum_PP(SP, std::vector< double > (NA) );
std::vector< std::vector< double > > sum_AA(SA, std::vector< double > (NA) );

std::vector<double> average_PP(SP);
std::vector<double> average_AA(SA);

double Sum_Peff, Sum_Aeff, Average_Peff, Average_Aeff;

std::vector< std::vector< double > > epsilon(SP, std::vector< double > (SA, 0.0) );
std::vector< std::vector< double > > new_epsilon(SP, std::vector< double > (SA) );

double alpha_P;
std::vector<double> alpha_A(SA);

std::vector<double> kappa(N);

std::vector< std::vector< double > > beta_P(SP, std::vector< double > (SP) );
std::vector< std::vector< double > > beta_A(SA, std::vector< double > (SA) );

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
double fl;
int fn;
double gamma_Plant;
double gamma_Pollinator;
double beta_Plant;
double beta_Pollinator;

std::vector<double> sigma(NS);
std::vector<double> alpha_average(N);

std::vector<double> RN(SA);

double t;

double rand_numa1, rand_numb1;

int main()
{
	std::ifstream fp;
	fp.open("linkage.txt");
	for(i=0;i<SP;i++)
	{
		for(j=0;j<SA;j++)
		{
			fp >> epsilon[i][j];
		}
	}
	fp.close();

/*===============================打印耦合矩阵===========================================
	for(i=0;i<SP;i++)
	{
		for(j=0;j<SA;j++)
		{
			printf("%lf ",epsilon[i][j]);
		}printf("\n");
	}
//=====================================================================================*/

	alpha_P=0.1;  mu=0.0001; tau=0.5; h=0.4; gamma_0=1.0; gc_beta=1.0;  //kappa=0.0;beta=1.0; alpha_A=-0.3;
	
	srand((unsigned)time(NULL));
	unsigned long init[4]={rand(), rand(), rand(), rand()},  length=4;
	//unsigned long init[4]={0x283, 0x175, 0x615, 0x954},length=4;
	init_by_array(init, length);
	
	for(i=0;i<SP;i++)
	{
		for(j=0;j<SP;j++)
		{
			if(j==i)
				beta_P[i][j]=1;
			else
				beta_P[i][j]=0;
		}
	}

	for(i=0;i<SA;i++)
	{
		for(j=0;j<SA;j++)
		{
			if(j==i)
				beta_A[i][j]=1;
			else
				beta_A[i][j]=0;
		}
	}
	
//===============计算节点的度===========================================================
	for(i=0;i<SP;i++)
	{
		K_n=0;
		for(j=0;j<SA;j++)
		{
			if(epsilon[i][j]==1) K_n+=1;
			else K_n=K_n;
		}
		K_P[i]=K_n;
	}
			
	for(j=0;j<SA;j++)
	{
		K_n=0;
		for(i=0;i<SP;i++)
		{
			if(epsilon[i][j]==1) K_n+=1;
			else K_n=K_n;
		}
		K_A[j]=K_n;
	}
/*===============================打印K_P[i]和K_A[i]======================================
	    for(i=0;i<SA;i++)
		{
			printf("%lf, ",K_A[i]);
		}
        printf("\n");

		for(i=0;i<SP;i++)
		{
			printf("%lf, ",K_P[i]);
		}
        printf("\n");
//=====================================================================================*/

/*===============================打印种内耦合矩阵===========================================
	for(i=0;i<SA;i++)
	{
		for(j=0;j<SA;j++)
		{
			printf("%lf ",beta_A[i][j]);
		}printf("\n");
	}
//=====================================================================================*/
   
	for(j_s=6;j_s<NS;j_s++)
	{
		sigma[j_s]=0.02*j_s;
		printf("=============sigma[%d]=%lf===================\n",j_s,sigma[j_s]);
		
		for(j_n=0;j_n<2;j_n++)
		{
			std::string fname = std::to_string(j_s) + "_Peff-Aeff_" + std::to_string(j_n+1) + ".csv";
			std::ofstream output_file(fname.c_str());
	
			for(j_ie=0;j_ie<N;j_ie++)
			{
				if(j_n==0)
					kappa[j_ie]=0.005*j_ie;
				else
					kappa[j_ie]=1.0-0.005*j_ie;
				
				for(j_a=0;j_a<NA;j_a++)
				{
	
	//=============================赋初值=========================================
					if(j_ie==0)
					{
						for(i=0;i<SP;i++)
						{
							P[i]=11+i*0.1; 
						}
						for(j=0;j<SA;j++)
						{
							A[j]=12+j*0.15; 
						}
					}
					else
					{
						for(i=0;i<SP;i++)
						{
							P[i]=init_P[i]; 
						}
						for(j=0;j<SA;j++)
						{
							A[j]=init_A[j]; 
						}
					}
					
		//============================================================================		
					for(j=0;j<SA;j++)
					{
						while(1)
						{
							rand_numa1 = genrand_real1(); rand_numb1 = genrand_real1();
							
							RN[j] = (0.1 + (sqrt(-2.0*log(rand_numa1)) * cos(2.0*pi*rand_numb1))*sigma[j_s]);
							if(RN[j]<1 && RN[j]>-1) break;
						}
						alpha_A[j] = RN[j];
					}
		//==============================================================================
					for(it=0;it<NT;it++)
					{
						t=it*dt;
						//printf("%lf\n",t);
		
						for(i=0;i<SP;i++)
						{
							temp_P[i]=P[i];
						}
					    for(j=0;j<SA;j++)
						{
							temp_A[j]=A[j];
						}
		//==============================================================================================================================================
						for(j=0;j<SA;j++)
						{
							gamma_Plant=0; beta_Pollinator=0;
							for(i=0;i<SP;i++)
								gamma_Plant+=epsilon[i][j]*gamma_0*temp_P[i]/(pow(K_A[j],tau));
		
							for(k=0;k<SA;k++)
								beta_Pollinator=beta_Pollinator+gc_beta*beta_A[j][k]*temp_A[k];
		
							A[j] = A[j] + ((alpha_A[j]-kappa[j_ie]-beta_Pollinator+(gamma_Plant/(1+h*gamma_Plant)))*A[j]+mu)*dt;
							//A[j]=A[j]+((alpha-A[j]-kappa[j_ie]+(gamma_Plant/(1+h*gamma_Plant)))*A[j]+mu)*dt;
						}
		
						for(i=0;i<SP;i++)
						{
							gamma_Pollinator=0; beta_Plant=0;
		
							for(j=0;j<SA;j++)
								gamma_Pollinator+=epsilon[i][j]*gamma_0*temp_A[j]/(pow(K_P[i],tau));
		
							for(k=0;k<SP;k++)
								beta_Plant=beta_Plant+gc_beta*beta_P[i][k]*temp_P[k];
		
							//printf("%lf %lf\n",beta_Plant,temp_P[i]);
		
							P[i]=P[i]+((alpha_P-beta_Plant+(gamma_Pollinator/(1+h*gamma_Pollinator)))*P[i]+mu)*dt;
							//P[i]=P[i]+((alpha-P[i]+(gamma_Pollinator/(1+h*gamma_Pollinator)))*P[i]+mu)*dt;
							//printf("%lf\n",(gamma_Pollinator/(1+h*gamma_Pollinator)));
						}
		//==================================================================================================================================================
					}   // loop of time
					
					Sum_P=0; Sum_A=0;
					
					for(i=0;i<SA;i++)
					{
						Sum_A+=A[i];
					}
					for(j=0;j<SP;j++)                                                                                                                                   
					{
						Sum_P+=P[j];
					}
					
					R_Peff[j_a]=Sum_P/SP;
					R_Aeff[j_a]=Sum_A/SA; 
					
					for(i=0;i<SA;i++)
					{
						init_A[i]=A[i];
					}
					for(j=0;j<SP;j++)                                                                                                                                   
					{
						init_P[j]=P[j];
					}
					
					for(i=0;i<SA;i++)
					{
						sum_AA[i][j_a] = A[i];
					}
					for(i=0;i<SP;i++)
					{
						sum_PP[i][j_a] = P[i];
					}
					
				}
				
				for(i=0;i<SP;i++)
				{
					average_PP[i] = 0.0;
				}
				
				for(i=0;i<SA;i++)
				{
					average_AA[i] = 0.0;
				}
				
				for(i=0;i<SP;i++)
				{
					for(j=0;j<NA;j++)
					{
						average_PP[i] += sum_PP[i][j] / NA ;
					}
				}
				
				for(i=0;i<SA;i++)
				{
					for(j=0;j<NA;j++)
					{
						average_AA[i] += sum_AA[i][j] / NA ;
					}
				}
				
				char out_char[100];
				if(j_ie%2==0)
				{
					sprintf(out_char, "%.10lf,",kappa[j_ie]);
					output_file << out_char;
					
					for(i=0;i<SP;i++)
					{
						sprintf(out_char, "%.10lf,",average_PP[i]);
						output_file << out_char;
					}
					
					for(i=0;i<SA;i++)
					{
						sprintf(out_char, "%.10lf,",average_AA[i]);
						output_file << out_char;
					}
				}
				
				Sum_Peff=0; Sum_Aeff=0; Average_Peff=0; Average_Aeff=0;
	
				for(i=0;i<NA;i++)
				{
					Sum_Aeff+=R_Aeff[i];
					Sum_Peff+=R_Peff[i];
				}
				Average_Aeff=Sum_Aeff/NA;
				Average_Peff=Sum_Peff/NA;
				
				if(j_ie%2==0)
				{
					printf("%lf\t%lf\t%lf\n",kappa[j_ie],Average_Peff,Average_Aeff);
					
					//char out_char[100];
					sprintf(out_char, "%.10lf,%.10lf\n",Average_Peff,Average_Aeff);
					output_file << out_char;
				}
				
			} //loop of N
			output_file.close();
		}
	} //loop of NS
	return 0; 
}

