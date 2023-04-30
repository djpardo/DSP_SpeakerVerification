function c = uniqueFeatureName(afeInfo)
    %UNIQUEFEATURENAME Create unique feature names
    %c = uniqueFeatureName(featureInfo) creates a unique set of feature names
    %for each element of each feature described in the afeInfo struct. The
    %afeInfo struct is returned by the info object function of
    %audioFeatureExtractor.
    a = repelem(fields(afeInfo),structfun(@numel,afeInfo));
    b = matlab.lang.makeUniqueStrings(a);
    d = find(endsWith(b,"_1"));
    c = strrep(b,"_","");
    c(d-1) = strcat(c(d-1),"0");
end