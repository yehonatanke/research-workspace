function [ImageOut]=error_diffusion(ImageIn)

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

close all
[high,width]=size(ImageIn); 
Error=zeros(high+1,width+1);   ImageOut=zeros(size(ImageIn));     ImageTmp=double(ImageIn);
for x=1:width,
        for y=1:high,
                if (ImageTmp(y,x)+Error(y,x))<128,
                    ImageOut(y,x)=0;
                else
                    ImageOut(y,x)=255;
                end;
                diff=(ImageTmp(y,x)+Error(y,x))-ImageOut(y,x);
                Error(y+1,x)= Error(y+1,x)+diff*3/8;
                Error(y,x+1)= Error(y,x+1)+diff*3/8;
                Error(y+1,x+1)= Error(y+1,x+1)+diff*1/4;
        end;
end;

%ImageOut=b;

if (nargout<1)
    montage ([ImageIn ImageOut]);
    title('error\_diffusion');
    clear ImageOut
end






