function report = audit_csv(csvFile)
%AUDIT_CSV Inspect a CSV recording without assuming EEG units or sample rate.
%   R = AUDIT_CSV(FILE) reports column types, missing values, likely EEG
%   channels, likely auxiliary/event fields, row count, and timestamp span
%   when a timestamp column can be parsed. It does not filter or rescale data.
%
%   The returned structure contains the imported table in R.data and printed
%   audit tables in R.columns and R.candidates.

arguments
    csvFile (1,1) string
end

if ~isfile(csvFile)
    error('audit_csv:FileNotFound', 'CSV file not found: %s', csvFile);
end

opts = detectImportOptions(csvFile, 'VariableNamingRule', 'preserve');
T = readtable(csvFile, opts);
if width(T) == 0
    error('audit_csv:Empty', 'CSV contains no columns.');
end

names = string(T.Properties.VariableNames(:));
rowCount = height(T);
columnCount = width(T);
kind = strings(columnCount,1);
missingCount = zeros(columnCount,1);
missingPercent = zeros(columnCount,1);
for k = 1:columnCount
    value = T.(T.Properties.VariableNames{k});
    kind(k) = string(class(value));
    missing = ismissing(value);
    missingCount(k) = sum(missing(:));
    if rowCount > 0
        missingPercent(k) = 100 * missingCount(k) / rowCount;
    end
end

% Recognize common scalp EEG labels, including labels followed by a reference
% or acquisition suffix (for example Fp1-REF or EEG_Fp1).
normalized = lower(regexprep(names, '[^a-zA-Z0-9]', ''));
channelPattern = '^(eeg)?(fp[12]|af[3578]|f[123478]|fz|fc[123456]|ft[78]|c[134]|cz|t[3578]|tp[78]|cp[123456]|p[134]|pz|po[3578]|o[12]|oz)(eeg)?(ref|le|re|avg|av|linkedears)?[0-9]*$';
likelyEEG = ~cellfun('isempty', regexp(cellstr(normalized), channelPattern, 'once'));

lowerNames = lower(names);
eventPattern = '(event|marker|trigger|stim|annotation|annot|label|condition|class|stress|state|trial|segment)';
timePattern = '(time|timestamp|date.?time|recorded.?at|elapsed)';
likelyEvent = ~cellfun('isempty', regexp(cellstr(lowerNames), eventPattern, 'once'));
likelyTime = ~cellfun('isempty', regexp(cellstr(lowerNames), timePattern, 'once'));

columnReport = table(names, kind, missingCount, missingPercent, ...
    'VariableNames', {'Column','DataType','MissingCount','MissingPercent'});
candidates = table(names(likelyEEG | likelyEvent | likelyTime), ...
    repmat("", sum(likelyEEG | likelyEvent | likelyTime), 1), ...
    'VariableNames', {'Column','CandidateRole'});
roles = strings(height(candidates),1);
for k = 1:height(candidates)
    j = find(names == candidates.Column(k), 1);
    labels = strings(0,1);
    if likelyEEG(j), labels(end+1) = "likely EEG"; end
    if likelyEvent(j), labels(end+1) = "possible event/annotation"; end
    if likelyTime(j), labels(end+1) = "possible timestamp"; end
    roles(k) = strjoin(labels, '; ');
end
candidates.CandidateRole = roles;

% Timestamp span is reported only when a named timestamp-like column parses
% unambiguously enough for MATLAB. No sampling frequency is inferred.
timeInfo = "No parseable timestamp column identified; duration cannot be established.";
parsedTimestampColumn = "";
spanSeconds = NaN;
for j = find(likelyTime(:))'
    v = T.(T.Properties.VariableNames{j});
    dt = NaT(rowCount,1);
    try
        if isdatetime(v)
            dt = v;
        elseif isnumeric(v)
            % Numeric timestamps have unknown units/epoch, so do not interpret.
            continue
        else
            dt = datetime(string(v));
        end
    catch
        continue
    end
    valid = dt(~isnat(dt));
    if numel(valid) >= 2
        spanSeconds = seconds(max(valid) - min(valid));
        parsedTimestampColumn = names(j);
        timeInfo = "Timestamp span from " + names(j) + ": " + string(spanSeconds) + " seconds.";
        break
    end
end

report = struct();
report.file = string(csvFile);
report.data = T;
report.rowCount = rowCount;
report.columnCount = columnCount;
report.columns = columnReport;
report.candidates = candidates;
report.likelyEEGChannels = names(likelyEEG);
report.possibleEventFields = names(likelyEvent);
report.possibleTimestampFields = names(likelyTime);
report.parsedTimestampColumn = parsedTimestampColumn;
report.timestampSpanSeconds = spanSeconds;
report.durationNote = timeInfo;

fprintf('CSV audit: %s\n', csvFile);
fprintf('Rows: %d | Columns: %d\n', rowCount, columnCount);
fprintf('Likely EEG channels (%d): %s\n', sum(likelyEEG), strjoin(names(likelyEEG), ', '));
fprintf('Possible event/annotation fields: %s\n', strjoin(names(likelyEvent), ', '));
fprintf('%s\n\n', timeInfo);
disp(columnReport);
if ~isempty(candidates)
    disp(candidates);
end
fprintf('Sampling frequency and physical units are not inferred by this audit.\n');
end

