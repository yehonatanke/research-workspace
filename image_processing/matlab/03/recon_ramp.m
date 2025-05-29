%function recon_ramp(n)
%%
n=20;
i=sqrt(-1);
x=0:999;
y=x;
Y=fft(y);
N=size(Y,2);
u=0;
recon=Y(u+1)*exp(2*i*pi*(u/N)*x)/N;
for u=1:n,
    recon=recon+(Y(u+1)*exp(2*i*pi*(u/N)*x)+Y(N-u+1)*exp(2*i*pi*(-u/N)*x))/N;
end;
close all
plot(x,y,'b',x,abs(recon),'r');