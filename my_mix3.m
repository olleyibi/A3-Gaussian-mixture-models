clc
clear all
close all

data = dlmread('old_faithful.dat','\t',26,0);
data = data(:,2:3);
X=data;

figure;
plot2 = gscatter(data(:,1), data(:,2));
hold on;
x1 = min(data(:,1))-2:0.1:max(data(:,1))+2;
x2 = min(data(:,2)):0.1:max(data(:,2));
[X1,X2] = meshgrid(x1,x2);


% initialization step
% declare parameters: all mean(s) and sigma(s), and P(s)
k_clusters = 2; % choose number of clusters
gam = zeros(size(data, 1), k_clusters);
means = zeros(k_clusters, size(data, 2)); % matrix of size k_clusters x D
P = ones(k_clusters, 1) / k_clusters;
sigmas = struct(); % structure that store covariance matrices of size D x D

% search for a good initialization values using
% some random sampled example
for ii = 1:k_clusters
    samples = data(randsample(1:size(data,1), 1), :);
    means(ii, :) = samples;
    sigmas(ii).covmat = 0.1 * rand(1,1) * cov(data);
    
    % plot the initial mixture models
    pdf_f = mvnpdf([X1(:) X2(:)],means(ii, :),sigmas(ii).covmat);
    pdf_f = reshape(pdf_f,length(x2),length(x1));
    [ct, plot1(ii)] = contour(x1,x2,pdf_f); 
    plot3(ii) = plot(means(ii,1),means(ii,2),'kx','LineWidth',2,'MarkerSize',10);
end

% EM n_iter steps
epsilon = 0.001;
n_iter = 1;
log_ll = [];
while(true)
    % Expectitation steps
    % Update gam
    for i = 1:size(data,1)
        % compute the new values of gam
        gamma_n = arrayfun(@(k) P(k) * mvnpdf(data(i,:), ...
                    means(k, :), sigmas(k).covmat), 1:k_clusters);
        % normalize them
        gamma_n = gamma_n / sum(gamma_n);
        gam(i, :) = gamma_n;
    end
    
    % Maximization steps
    % Estimate mean(s), sigma(s), and covariance matrix
    Ns = sum(gam);
    for j = 1:k_clusters
        % update j-th mean
        for d = 1:size(data,2) % iterate through data dimension
            means(j, d) = sum(arrayfun(@(i) gam(i, j) * ...
                            data(i, d), 1:size(data, 1)))/Ns(j);
        end
        
        % update j-th covariance matrix
        sigmas(j).covmat = ((data(:,:)-means(j,:))' * ...
                            (gam(:, j) .* (data(:,:)-means(j,:))))/Ns(j);
        
        % update j-th P
        P = (Ns/sum(Ns))';
        
        delete(plot1(j));
        delete(plot2);
        delete(plot3(j));
        
        [M,I] = max(gam, [], 2);
        plot2 = gscatter(data(:,1), data(:,2), I);
        
        pdf_f = mvnpdf([X1(:) X2(:)],means(j, :),sigmas(j).covmat);
        pdf_f = reshape(pdf_f,length(x2),length(x1));
        [ct, plot1(j)] = contour(x1,x2,pdf_f); 
        
        plot3(j) = plot(means(j,1),means(j,2),'kx','LineWidth',2,'MarkerSize',10);
        uistack(plot2, 'bottom');
        drawnow;
    end
    
    % compute log likelihood
    % plot the progress
    accum_L = 0;
    for i = 1:size(data,1)
        i_accum_L = 0;
        for j = 1:k_clusters
            i_accum_L = i_accum_L + P(j)* mvnpdf(data(i,:), means(j, :), sigmas(j).covmat);
        end
        accum_L = accum_L + log(i_accum_L);
    end
    log_ll = [log_ll; accum_L];
    n_iter = n_iter + 1;
    
    % stopping condition
    % case when log-likelihood stop improving by more than eps
    if numel(log_ll) > 1
        if abs(log_ll(end)-log_ll(end-1)) <= epsilon
            fprintf('[Optimization completed]\n');
            break;
        end
    end
end