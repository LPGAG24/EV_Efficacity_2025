function [] = Write_Matrix(path_, data)
%   Write binary mat file
    [m, n] = size(data);
    fp = fopen(path_, 'w');
    fwrite(fp, m, 'int');
    fwrite(fp, n, 'int');
    fwrite(fp, data, 'double');
    fclose(fp);
end