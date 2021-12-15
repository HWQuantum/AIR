function VMIisosurf(I_3D,cont,shape)

% Shows 3D VMI data with a series isosurfaces of different colours to
% indicate electron intensity.
%
% The colors can be edited by the user bellow

colors = [0 0 0.5; 0 0 1; 0.5 0.5 1; 1 1 1; 1 0.5 0.5; 1 0 0; 0.5 0 0]; % blue/red map

% Write map
%
% This step creates a colormap using the above colors and interpolates
% between them to create any number of desired isosurfaces

int_step = 12;

map = [];

for i = 1:3
    
    x = [];
    
    for n = 1:size(colors,1)-1
        
        x_new = linspace(colors(n,i),colors(n+1,i),int_step);
        
        if n ~= size(colors,1)-1
            
            x_new = x_new(1,1:end-1);
            
        else
        
        end
        
        x = [x, x_new];
        
    end
    
    map = [map; x];
    
end

map = map';

% Create the voxel grid and details

steps = size(map,1); % the number of different isosurfaces to be calculated

val = linspace(1,-1,steps); % defines surface values

I_3D = imresize3(I_3D,70*[1,1,1]); % rescales input resolution (optional)

res = size(I_3D,1)/2;

x = linspace(-1,1,2*res);

[X, Y, Z] = meshgrid(x, x, x); % create voxel coordinate space

alpha = 0.5; % how transparent to make surfaces

norm = max(max(max(abs(I_3D))));

% Build the distribution

figure

hAxis=axes;

if strcmp(shape,'split') == 1 % creates two hemispheres, seperated along x-axis
    
    splitting = 2;
    
    for i = 1:steps
        
        hold on
        
        % Creates an isosurface for each val and gives it the corresponding
        % color in map
    
        [F, V] = isosurface(X(:,1:35,:)-splitting/2, Y(:,1:35,:), Z(:,1:35,:), I_3D(:,1:35,:), (1/cont)*norm*val(1,i));
    
        p1 = patch('Faces', F, 'Vertices', V);
        p1.FaceColor = sign(map(steps-i+1,:)).*map(steps-i+1,:);
        p1.FaceAlpha = abs(val(1,i)).^(alpha);
        p1.EdgeColor = 'none';
        
    end
    
    for i = 1:steps
        
        hold on
    
        [F, V] = isosurface(X(:,36:end,:)+splitting/2, Y(:,36:end,:), Z(:,36:end,:), I_3D(:,36:end,:), (1/cont)*norm*val(1,i));
    
        p1 = patch('Faces', F, 'Vertices', V);
        p1.FaceColor = sign(map(steps-i+1,:)).*map(steps-i+1,:).^2;
        p1.FaceAlpha = abs(val(1,i)).^(alpha);
        p1.EdgeColor = 'none';
        
    end
    
    hold off
    
    xlim([-2 2])
    ylim([-1 1])
    zlim([-1 1])

    pbaspect([2 1 1])
      
else

    for i = 1:steps
    
        hold on
    
        [F, V] = isosurface(X, Y, Z, I_3D(:,:,:), (1/cont)*norm*val(1,i));

        p1 = patch('Faces', F, 'Vertices', V);

        p1.FaceColor = sign(map(steps-i+1,:)).*map(steps-i+1,:);
        p1.FaceAlpha = abs(val(1,i)).^(alpha);
        p1.EdgeColor = 'none';
       
    end
    
end

hold off

view([-50 25])
if strcmp(shape,'half') == 1 % shows half of distribution (recommended)
    
    xlim(1.2*[0 1])
    ylim(1.2*[-1 1])
    zlim(1.2*[-1 1])
    pbaspect([0.5 1 1])
    
elseif strcmp(shape,'full') == 1 % shows full distribution (not recommended)
    
    xlim([-1 1])
    ylim([-1 1])
    zlim([-1 1])
    pbaspect([1 1 1])
    
else
    
end

hAxis.ZAxis.FirstCrossoverValue  = 0; 
hAxis.ZAxis.SecondCrossoverValue = 0;

hAxis.YAxis.FirstCrossoverValue  = 0; 
hAxis.YAxis.SecondCrossoverValue = 0;

hAxis.XAxis.FirstCrossoverValue  = 0; 
hAxis.XAxis.SecondCrossoverValue = 0;

set(gca,'XMinorTick','on','YMinorTick','on','ZMinorTick','on','LineWidth',1)

xticklabels([])
yticklabels([])
zticklabels([])

colormap(map)
caxis((cont)*norm*[-1 1])


end

