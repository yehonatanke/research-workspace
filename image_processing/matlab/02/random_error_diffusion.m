function [ImageOut]=random_error_diffusion(ImageIn)

if (nargin<1)
    DialogTitle='Input file name';
    FilterSpec={'*.jpg'; '*.png' ; '*.tif' ;'*.gif'};
    DefaultName='lena.gif';
    [FileName,PathName,FilterIndex] =uigetfile(FilterSpec,DialogTitle,DefaultName) ;
    if ( isequal(FilterIndex,0) )
   disp('User selected Cancel')
   return;
    else
    ImageFullName=fullfile(PathName, FileName);
   disp(['User selected:  ', ImageFullName]);
    ImageIn=imread(ImageFullName);
    end
end

figure; %close all  %

[high,width]=size(ImageIn); 
Error=zeros(high+1,width+1);   ImageOut=zeros(size(ImageIn));     ImageTmp=double(ImageIn);
Prob1=rand(size(ImageIn));
Prob2=rand(size(ImageIn));
Prob3=rand(size(ImageIn));
ProbS=Prob1+Prob2+Prob3;
ind=find(ProbS==0);
Prob1(ind)=1;
Prob2(ind)=1;
Prob3(ind)=1;
ProbS(ind)=3;
Prob1=Prob1./ProbS;
Prob2=Prob2./ProbS;
Prob3=Prob3./ProbS;

for x=1:width,
        for y=1:high,
                if (ImageTmp(y,x)+Error(y,x))<128,
                    ImageOut(y,x)=0;
                else
                    ImageOut(y,x)=255;
                end;
                diff=(ImageTmp(y,x)+Error(y,x))-ImageOut(y,x);
                Error(y+1,x)= Error(y+1,x)+diff*Prob1(y,x);
                Error(y,x+1)= Error(y,x+1)+diff*Prob2(y,x);
                Error(y+1,x+1)= Error(y+1,x+1)+diff*Prob3(y,x);
        end;
end;


if (nargout<1)
    montage ([ImageIn ImageOut]);
    title('random\_error\_diffusion');
    clear ImageOut
end






