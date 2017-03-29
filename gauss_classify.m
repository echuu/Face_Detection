function [h_vector, lpp_ratio] = gauss_classify(x, m1, s1, m0, s0)

centered_1 = (x - m1).^2 / s1.^2;
centered_0 = (x - m0).^2 / s0.^2;

lpp_ratio = - 1/2 .* (centered_1 - centered_0 + log(s1) - log(s0));

h_vector = -ones(size(lpp_ratio));
positive_lpp = lpp_ratio > 0;
h_vector(positive_lpp) = 1;