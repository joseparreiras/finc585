/*
Author: Jose Antunes-Neto

This code contains functions to download Consolidated Trades data from NYSE's TAQ available on WRDS. The TAQ database contains a structural change starting on 2015. For this reason, I create 2 functions to pull the observations, one for each period. Each of these functions take as arguments the SYMBOL of the stock that should be downloaded, the DATE and the FREQUENCY in seconds that the data should be returned. Finally, I create a function that iterates the data extraction and appends them all at the end. For each day, the macros detect whether a dataset exists for that given date. If not, a message is returned on the log, but the code keeps going.
*/

options mprint mlogic symbolgen; * Set options to print macro variables and logics;
proc datasets library=work kill nolist; run; * Clear the work library;

%macro extract_post2015(symbol, dt, freq);
	/* 
	This first macro extracts the data from the TAQ database for the post-2015 period. The data is stored in the taqmsec library. The data is filtered by the symbol, time, and trade condition. The data is aggregated by the datetime and the average price is calculated. Price is calculated as the average traded price over the period, weighted by the size of the trades. Data is aggregated by datetime and stored in the work library.

	Arguments:
		SYMBOL: Stock symbol;
		DT: Date (yyyymmdd);
		FREQ: Frequency in seconds;
	*/
	%if %sysfunc(exist(taqmsec.ctm_&dt)) %then %do; * Check if the dataset exists;
		/* Define variables */
	    %let d = &dt; * Date;
		%let f = taqmsec.ctm_&dt; * Source library;
		%let dbname = &symbol._&dt; * Name of final dataset;
		%let freq = &freq; * Frequency in second;
	
		/* Extract initial data from TAQMSEC */
		proc sql;
			CREATE TABLE temp AS
			SELECT DATE, PRICE, SIZE, SYM_ROOT as SYMBOL, TIME_M as TIME * Rename symbol and time to match the old TAQ database;
			FROM &f
			WHERE SYMBOL = "&symbol" * Filter by symbol;
			and TIME between "09:30:00"t and "16:00:01"t * Filter by time;
			and SIZE > 0 * Filter by size;
			and TR_CORR = "00"; * Remove canceled trades;
		run;

		/* Create a DATETIME column on the temporary dataset */
		data temp;
			set temp;
			TIME = TIME - mod(TIME,&freq); * Round the time to the frequency;
    		DATETIME = DHMS(DATE, HOUR(TIME), MINUTE(TIME), SECOND(TIME)); * Create the datetime variable;
			format TIME TIME.; * Format the time variable;
			format DATETIME datetime21.; * Format the datetime variable;
		run;
	
		/* Aggregate the data by datetime averaging the price by size*/
		proc sql;
			create table &dbname as
			select DATETIME, SYMBOL, round(sum(PRICE * SIZE)/sum(SIZE),  0.01) as PRICE * Calculate the average price (round to 2 decimals);
			from temp
			group by DATETIME * Group and order by datetime;
			order by DATETIME;
		run;
				
		/* Keep only the first observation of each datetime (Remove duplicates)*/
		data &dbname;
			set &dbname;
			by DATETIME;
			if first.DATETIME;
		run;
		
		/* Delete the temporary dataset */
		proc datasets library=work; delete temp; run;
  	%end; 
   
 	%else %do;
		/* If the dataset does not exist, print a message on the log */
		data _null_;
			file print;
			put #3 @10 "Data set &dt. does not exist";
		run;
	%end;
%mend extract_post2015;

%macro extract_pre2015(symbol, dt, freq);
	/* 
	This first macro extracts the data from the TAQ database for the pre-2015 period. The data is stored in the taq library. The data is filtered by the symbol, time, and trade condition. The data is aggregated by the datetime and the average price is calculated. Price is calculated as the average traded price over the period, weighted by the size of the trades. Data is aggregated by datetime and stored in the work library.

	Arguments:
		SYMBOL: Stock symbol;
		DT: Date (yyyymmdd);
		FREQ: Frequency in seconds;
	*/
	%if %sysfunc(exist(taq.ct_&dt)) %then %do; * Check if the dataset exists;
		/* Define variables */
		%let d = &dt; * Date;
		%let f = taq.ct_&d; * Source library;
		%let dbname = &symbol._&dt; * Name of final dataset;
		%let freq = &freq; * Frequency in seconds;
		
		/* Extract initial data from TAQ */
		proc sql;
			CREATE TABLE temp AS
			SELECT DATE, PRICE, SIZE, TIME, SYMBOL
			FROM &f
			WHERE SYMBOL = "&symbol" * Filter by symbol;
			and TIME between "09:30:00"t and "16:00:01"t * Filter by time;
			and SIZE > 0 * Filter by size;
			and CORR = 0; * Remove canceled trades;
		run;
		
		/* Create a DATETIME column on the temporary dataset */
		data temp;
			set temp;
			TIME = TIME - mod(TIME,&freq); * Round the time to the frequency;
			DATETIME = DHMS(DATE, HOUR(TIME), MINUTE(TIME), SECOND(TIME)); * Create the datetime variable;
			format TIME TIME.; * Format the time variable;
			format DATETIME datetime21.; * Format the datetime variable;
		run;
	
		/* Aggregate the data by datetime averaging the price by size */
		proc sql;
			create table &dbname as
			select DATETIME, SYMBOL, round(sum(PRICE * SIZE)/sum(SIZE),  0.01) as PRICE * Calculate the average price (round to 2 decimals);
			from temp
			group by DATETIME * Group and order by datetime;
			order by DATETIME;
		run;
	
		/* Keep only the first observation of each datetime (Remove duplicates) */
		data &dbname;
			set &dbname;
			by DATETIME;
			if first.DATETIME;
		run;
	
		/* Delete the temporary dataset */
		proc datasets library=work; delete temp; run;
   %end;
   
   /* If the dataset does not exist, print a message on the log */
   %else %do;
      data _null_;
         file print;
         put #3 @10 "Data set &dt. does not exist";
      run;
   %end;
%mend extract_pre2015;

%macro extract_symbol(symbol, start_date, end_date, freq);
	/* 
	Iterates extraction of data for a given symbol. For dates prior to 2015, it uses the extract_pre2015 function. For dates after 2015, it uses the extract_post2015 function. The data is appended to the symbol dataset.
	
	Arguments:
		SYMBOL: Stock symbol;
		START_DATE: Start date (yyyymmdd);
		END_DATE: End date (yyyymmdd);
		FREQ: Frequency in seconds;
	*/
	/* Format start and end dates */
    %let start_date = %sysfunc(inputn(&start_date, yymmddn8.));
    %let end_date = %sysfunc(inputn(&end_date, yymmddn8.));

	/* Loop */
    %do day=&start_date %to &end_date;
		/* Format the date to a number (used in the dataset name) */
    	%let day_num = %sysfunc(inputn(&day, 8.));
        %if &day ge 20150101 %then %do; * Check if the date is after 2015;
            %extract_post2015(&symbol, &day, &freq); * Extract the data for the post-2015 period;
        %end;
        %else %do; * If the date is before 2015;
            %extract_pre2015(&symbol, &day, &freq); * Extract the data for the pre-2015 period;
        %end;

        /* Append the result to the symbol dataset. It first checks whether the final data already exists. If so, it uses `proc append` to concatenate both datasets. If not, a new dataset is created with the first iteration result.*/
        %if %sysfunc(exist(&symbol._&day)) %then %do;
            %if %sysfunc(exist(&symbol)) %then %do; * Check if the dataset exists;
                proc append base=&symbol data=&symbol._&day_num force; * Append the data;
            %end;
            %else %do; * If the dataset does not exist;
                data &symbol;
                    set &symbol._&day_num; * Create the dataset;
                run;
            %end;
        %end;
        
		/* Delete the temporary daily dataset */
        proc datasets library=work; delete &symbol._&day_num; run;
    %end;
%mend extract_symbol;

%macro export_zip(data, outdir, fname);	
	/*
	This last function is used to extract large datasets into a zip file. It exports the data to a temporary csv file and then creates a zip file with the csv file. The csv file is deleted after the zip file is created.

	Arguments:
		DATA: Dataset to be exported;
		OUTDIR: Output directory;
		FNAME: Name of the zip file (without the extension);
	*/
	%let fcsv = "&outdir./&fname..csv"; * Temporary csv file;
	
	/* Export the dataset table to a temporary csv file */
	proc export data = &data outfile = &fcsv dbms = csv replace; run; 

	/* Create the zip file */
	data _null_; 
		/* Creates a new zipfile and add temporary csv file */
	    ods package(myout_zip) open nopf; 
	    	ods package(myout_zip) add file = &fcsv; * Add the csv file to the zip file;
	    	ods package(myout_zip) publish archive properties(
	    		archive_name = "&fname..zip" archive_path = "&outdir"
	    		);
		ods package(myout_zip) close; * Close the zip file;
		/* Renames and delete temporary csv file */
		rc = filename("csvfile", &fcsv); * Rename the csv file;
		rc = fdelete("csvfile"); * Delete the csv file;
	run;
%mend export_zip;

/* Example of usage */
%let startdate = 19930101; * Start date;
%let enddate = 20240331; * End date;
%let freq = 180; * Frequency in seconds;
%let symbol = SPY; * Stock symbol;
%let outdir = /path/to/output; * Output directory;

%extract_symbol(&symbol, &startdate, &enddate, &freq); 
%export_zip(&symbol, &outdir, &symbol); * Export the dataset to a zip file;
/*  */

quit;