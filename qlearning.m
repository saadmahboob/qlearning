   % Q learning
   % Two input: R and gamma
   % immediate reward matrix; 
   % row(depart) and column(target) = states; -Inf = no door between room
   % The goal room is F, so the doors that lead immediately to the goal have instant reward of 100 
   %   A   B   C   D   E   F
   % A                 1  
   % B             1       1
   % C             1
   % D     1   1       1
   % E 1           1       1
   % F     1           1   1


function qlearning
clear all;close all; clc;
    R=[ -inf,   -inf,   -inf,   -inf,    0,      -inf;
        -inf,   -inf,   -inf,   0,      -inf,   100;
        -inf,   -inf,   -inf,   0,      -inf,   -inf;
        -inf,   0,      0,      -inf,   0,      -inf;
        0,      -inf,   -inf,   0,      -inf,   100;
        -inf,   0,      -inf,   -inf,   0,      100];
labels=['A' 'B' 'C' 'D' 'E' 'F'];
gamma=0.80;            % learning parameter
nEpisodes=5000;
[A episode]=ReinforcementLearning(R,nEpisodes,gamma);
adj=(A>0);

%# plot adjacency matrix
subplot(121), spy(adj)
title('Adjacency matrix');
%# plot connected points on grid
X=[1 2 3.5 2 1 1.5]';
Y=[2 2 1.5 1 1 1.5]';
[xx yy] = gplot(adj, [X Y]);
subplot(122), plot(xx, yy, 'ks-', 'MarkerFaceColor','r')
title('Graph');
axis([0.5 4 0.5 2.5])
dx=0.1;dy=0.1;

for j=1:length(labels)
    text(X(j)+dx,Y(j)+dy,labels(j))
end


%% Find path
initial=3;  %C
goal=6;     %F

foundPath=false;
current=initial;

index=1;
pathPlan(index)=current;
while(not(foundPath))
    [val current]=max(A(current,:));
    index=index+1;
    pathPlan(index)=current;
    if(current==goal)
        foundPath=true;
    end
end
disp('Path:')
for j=1:length(pathPlan)
   disp(sprintf('%c',labels(pathPlan(j))))
end
end

 function y=RandomPermutation(A)
 % return random permutation of matrix A
 % unlike randperm(n) that give permutation of integer 1:n only
   [r,c]=size(A);
   b=reshape(A,r*c,1);       % convert to column vector
   x=randperm(r*c);          % make integer permutation of similar array as key
   w=[b,x'];                 % combine matrix and key
   d=sortrows(w,2);          % sort according to key
   y=reshape(d(:,1),r,c);    % return back the matrix
 end
 
 function [q episode]=ReinforcementLearning(R, nepisodes,gamma)
format short
format compact
    q=zeros(size(R));      % initialize Q as zero
    q1=ones(size(R))*inf;  % initialize previous Q as big number
    count=0;               % counter

    for episode=0:nepisodes
       % random initial state
       y=randperm(size(R,1));
       state=y(1);
       
       % select any action from this state
       x=find(R(state,:)>=0);        % find possible action of this state
       if size(x,1)>0,
          x1=RandomPermutation(x);   % randomize the possible action
          x1=x1(1);                  % select an action 
       end

       qMax=max(q,[],2);
       q(state,x1)= R(state,x1)+gamma*qMax(x1);   % get max of all actions 
       state=x1;
       
       % break if convergence: small deviation on q for 1000 consecutive
       if sum(sum(abs(q1-q)))<0.0001 & sum(sum(q >0))
          if count>1000,
             episode        % report last episode
             break          % for
          else
             count=count+1; % set counter if deviation of q is small
          end
       else
          q1=q;
          count=0; % reset counter when deviation of q from previous q is large
       end
    end 

    %normalize q
    g=max(max(q));
    if g>0, 
       q=100*q/g;
    end
end
      

