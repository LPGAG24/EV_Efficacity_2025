function [data] = Read_Matrix(path_)
%   Read binary mat file
    fp = fopen(path_, 'r');
    m = fread(fp, 1, 'int');
    n = fread(fp, 1, 'int');
    data = fread(fp, m * n, 'double');
    fclose(fp);
    
    data = reshape(data, [m, n]);
end