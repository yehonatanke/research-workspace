function  pseodo_color(ImageIn)

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


close all  % figure; %
N=1024;
sigma1=500;
sigma2=300;
sigma3=200;
mycolormap=zeros(N,3);
mycolormap(N:-1:1,1)=exp(-(0:N-1)/sigma1);
mycolormap(N:-1:1,2)=exp(-(0:N-1)/sigma2);
mycolormap(N:-1:1,3)=exp(-(0:N-1)/sigma3);

IndexImageIn=floor(N*double(ImageIn)/255);
figure; imshow (IndexImageIn,mycolormap);
title('pseodo\_color');


IndexImageIn=floor(N*double(ImageIn)/255);
figure; imshow (IndexImageIn,hot(N));
title('pseodo\_color');




