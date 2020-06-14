// Code for T test statistics for Granger causality test
// 23/04/2005

// optimized for Linux

// for comments/suggestions please contact Valentyn Panchenko v.panchenko@uva.nl

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define max(a,b)	a>b?a:b

double T2,*h;

//returns T2 statistics
void redun(double *x, double *y, int N, int m, int mmax, double
epsilon)
{

  int i, j, s;
  int IYij, IXYij, IYZij, IXYZij;
  double disx, disy, disz, *Cy, *Cxy, *Cyz, *Cxyz, mu;

  mu=pow(2.0*epsilon,m+2*mmax+1);


  Cy = (double *) malloc(N*sizeof(double));
  Cxy = (double *) malloc(N*sizeof(double));
  Cyz = (double *) malloc(N*sizeof(double));
  Cxyz = (double *) malloc(N*sizeof(double));


  for (i=0;i!=N;i++)
    h[i] = Cy[i] = Cxy[i] = Cyz[i] = Cxyz[i] = 0.0;

  T2=0.0;

  for (i=mmax;i!=N;i++)
  {
    Cy[i]=Cxy[i]=Cyz[i]=Cxyz[i]=0.0;
    for (j=mmax;j!=N;j++)
    if (j!=i)

    {                                               
      disx = disy = 0.0;
      for (s=1;s!=m+1;s++)
        disx = max(fabs(x[i-s]-x[j-s]),disx);

      for (s=1;s!=mmax+1;s++)
        disy = max(fabs(y[i-s]-y[j-s]),disy);

      if (disy <= epsilon)
      {
	Cy[i]++;

        if (disx <= epsilon)
        {
          Cxy[i]++;
        }

        disz = max(fabs(y[i]-y[j]),disy);
        if (disz <= epsilon)
        {
          Cyz[i]++;
          if (disx <= epsilon)
          {
            Cxyz[i]++;
          }
        }
      }   // end condition |Yi - Yj| < epsilon
    }   // end loop over j

    Cy[i] /= (double)(N-mmax);
    Cxy[i] /= (double)(N-mmax);
    Cyz[i] /= (double)(N-mmax);
    Cxyz[i] /= (double)(N-mmax);

    h[i] += 2.0/(double) mu*(Cxyz[i]*Cy[i] - Cxy[i]*Cyz[i])/6.0;

  }

  for (i=mmax;i!=N;i++)
  {
    for (j=mmax;j!=N;j++)
    if (j!=i)
    {

      IYij = IXYij = IYZij = IXYZij = 0;
      disx = disy = 0.0;

      for (s=1;s!=m+1;s++)
        disx = max(fabs(x[i-s]-x[j-s]),disx);

      for (s=1;s!=mmax+1;s++)
        disy = max(fabs(y[i-s]-y[j-s]),disy);

      if (disy <= epsilon)
      {

        IYij=1;
        if (disx <= epsilon)
	  IXYij = 1;

        disz = max(fabs(y[i]-y[j]),disy);
        if (disz <= epsilon)
        {
	  IYZij = 1;
          if (disx <= epsilon)
            IXYZij = 1;
        }
      }   // end condition |Yi - Yj| < epsilon

      h[i] += 2.0/(double) mu*(Cxyz[j]*IYij + IXYZij*Cy[j] - Cxy[j]*IYZij - IXYij*Cyz[j])/(double)(6*(N-mmax));
    }   // end second loop over j
  } // end loop over i

  for (i=mmax;i!=N;i++)
    T2 += h[i];

  T2 /= (double)(N-mmax);
  for (i=mmax;i!=N;i++)
     h[i] -= T2;



  free (Cy);
  free (Cxy);
  free (Cxyz);
  free (Cyz);

}

void InsertionSort(double *X, int *S, int M)
{
    int i, *I;
    int j;
    int r;
    double R;

    I= (int*) malloc (M*sizeof(int));

    for (i=0;i<M;i++)
      I[i]=i;

    for (i=1; i<M; i++)
      {
        R = X[i];
        r = i;
	for (j=i-1; (j>=0) && (X[j]>R); j--)
        {
	  X[j+1] = X[j];
          I[j+1] = I[j];
        }
	X[j+1] = R;
        I[j+1] = r;
      }
    for (i=0; i<M; i++)
      S[I[i]]=i;

}


void  uniform (double *X, int M)
{
  int *I, i;

  I = (int*) malloc (M*sizeof(int));
  InsertionSort(X, I, M);

  for (i=0;i<M;i++)
    X[i] = (double) I[i]/M*3.464101615;        // to make unit variance

}


/* normalize the time series to unit std. dev. */

void normalise(double *x, int N)
{
  int i;
  double mean=0.0, var=0.0;

  for (i=0;i!=N;i++)
  {
    mean += x[i];
    var += x[i]*x[i];
  }

  mean /= (double)(N);
  var /= (double)(N);
  var -= mean*mean;

  for (i=0;i!=N;i++)
    x[i] = (x[i]-mean)/sqrt(var);

  return;
}

int main(int num_par, char *par[])
{
  char infil1name[128]="test1.txt",infil2name[128]="test2.txt", outfilname[128];
  double *x, *y, tmp, epsilon=.50, VT2,  p_T2, p_T21, *ohm, *cov, T2_TVAL, T2_TVAL1, sigma[4][4];
  int i, j, l, k, m=1, K, N;
  long seed;
  FILE *infil1, *infil2, *outfil;

    // enter parameters from outside
  if (num_par==1)
  {
    printf("Input file containing series 1: ");
    scanf("%s", infil1name);

  }
  else
  {
    i=0;
    do
    {
      infil1name[i]=par[1][i];
      i++;
    }
    while (par[1][i-1]!='\0');
  }


  if ( (infil1=fopen(infil1name,"r")) == NULL)
  {
    fprintf(stderr,"\nError: unable to open file containing series 1...%s\n",infil1name);
    exit(1);
  }

  i = 0;

  while (fscanf(infil1,"%lf", &tmp) != EOF)
  {
    i++;
  }
  fclose(infil1);

  N=i;

  if (num_par<3){
    printf("Input file containing series 2: ");
    scanf("%s", infil2name);
  }
  else
  {
    i=0;
    do
    {
      infil2name[i]=par[2][i];
      i++;
    }
    while (par[2][i-1]!='\0');
  }

  if ( (infil2=fopen(infil2name,"r")) == NULL)
  {
    fprintf(stderr,"\nError: unable to open file containing series 2.\n");
    exit(1);
  }

  i=0;
  while (fscanf(infil2,"%lf", &tmp) != EOF)
  {
    i++;
  }
  fclose(infil2);

  if ( i!=N)
  {
    fprintf(stderr,"\nError: files contain series of different length.\n");
  }

  if (num_par<4)
  {
    printf("Input embedding dimension: ");
    scanf("%d", &m);
  }
  else
  m=atoi(par[3]);

  if (num_par<5)
  {
    printf("Input bandwidth: ");
    scanf("%lf", &epsilon);
    printf("\n");
  }
  else
    epsilon=atof(par[4]);


  x = (double *) malloc(N*sizeof(double));
  y = (double *) malloc(N*sizeof(double));

  h = (double *) malloc(N*sizeof(double));

  K = (int)(sqrt(sqrt(N)));
  ohm = (double *) malloc(K*sizeof(double));
  cov = (double *) malloc(K*sizeof(double));

// read the series
  infil1=fopen(infil1name,"r");
  infil2=fopen(infil2name,"r");

  for (i=0;i<N;i++)
  {
    fscanf(infil1,"%lf",&(x[i]));
    fscanf(infil2,"%lf",&(y[i]));
  }

  normalise(x,N);
  normalise(y,N);

  // redun(x,y, ..) test statistic for X -> Y
  redun(x,y,N,m,m,epsilon);

  ohm[0] = 1.0;

  for (k=1;k<K;k++)
    ohm[k] = 2.0*(1.0-k/(double)(K));

  /* determine autocovariance of h[i] */

  for (k=0;k!=K;k++)
  {
    cov[k] = 0.0;
    for (i=m+k;i!=N;i++)
      cov[k] += h[i]*h[i-k];

    cov[k] /= (double)(N-m-k);
  }

  T2_TVAL=VT2=0.0;

/* variance of T2 */

  for (k=0;k!=K;k++)
    VT2 += 9.0*ohm[k]*cov[k];

  T2_TVAL = T2*sqrt(N-m)/sqrt(VT2);

  p_T2 = 0.5 - .5*erf(T2_TVAL/sqrt(2.0));
 
  if ((num_par<6) || ((outfil=fopen(par[5],"w")) == NULL)) outfil=stdout;
  else printf("The results are saved to the file: %s\n",par[5]);
  fprintf(outfil,"Series length=%d, embedding dimension=%d, bandwidth=%f\n",N,m,epsilon);
  fprintf(outfil,"\nNull hypothesis: %s does not cause %s\n",infil1name,infil2name);
  fprintf(outfil,"T statistics=%.3f, p-value=%1.5f\n",T2_TVAL,p_T2);

  redun(y,x,N,m,m,epsilon);

  ohm[0] = 1.0;

  for (k=1;k<K;k++)
    ohm[k] = 2.0*(1.0-k/(double)(K));

  /* determine autocovariance of h[i] */

  for (k=0;k!=K;k++)
  {
    cov[k] = 0.0;
    for (i=m+k;i!=N;i++)
      cov[k] += h[i]*h[i-k];

    cov[k] /= (double)(N-m-k);
  }

  T2_TVAL=VT2=0.0;

/* variance of T2 */

  for (k=0;k!=K;k++)
    VT2 += 9.0*ohm[k]*cov[k];

  T2_TVAL = T2*sqrt(N-m)/sqrt(VT2);

  p_T2 = 0.5 - .5*erf(T2_TVAL/sqrt(2.0));

  fprintf(outfil,"\nNull hypothesis: %s does not cause %s\n",infil2name,infil1name);
  fprintf(outfil,"T statistics=%.3f, p-value=%1.5f\n",T2_TVAL,p_T2);

  fcloseall();
  return(0);
}

