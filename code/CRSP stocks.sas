/* 
   Purpose: 
   This SAS code retrieves stock data for the current constituents of the S&P 500 index 
   for Principal Component Analysis (PCA) of returns.

   Steps:
   1. Create a table of stocks (`stocks`) by selecting data from the CRSPA dataset `dsf`.
   2. Select stocks from the S&P 500 List (`selected_stocks`) based on the latest ending date.
   3. Filter the final stocks (`stocks_final`) based on selected stocks and sort by `permno` and `date`.
   4. Export the final dataset to a ZIP file for further analysis.
*/

/* Enable SAS options for debugging */
options mprint mlogic symbolgen;

/* Clear existing datasets from the work library */
proc datasets library=work kill nolist;

/* Create table of stocks */
proc sql;
    create table stocks as
    select date format date9., permno, abs(prc)*shrout as market_cap, round(abs(prc),0.01) as price
    from crspa.dsf
    where shrout is not null
        and prc is not null
        and date >= "01JAN2020"d;
quit;

/* Select stocks from S&P 500 List */
proc sql;
    create table selected_stocks as
    select permno
    from crsp.dsp500list
    where ending = (select max(ending) from crsp.dsp500list);
quit;

/* Filter final stocks */
proc sql;
    create table stocks_final as
    select date, permno, price
    from stocks
    where permno in (select distinct permno from selected_stocks)
    order by permno, date asc;
quit;

/* Macro to export dataset to ZIP */
%macro export_zip(data, outdir, fname);  
    /* Define file paths */
    %let fcsv = "&outdir./&fname..csv"; 
    
    /* Export dataset to CSV */
    proc export data = &data outfile = &fcsv dbms = csv replace; 
    run; 

    /* Create ZIP file and add CSV */
    data _null_; 
        ods package(myout_zip) open nopf; 
            ods package(myout_zip) add file = &fcsv;
            ods package(myout_zip) publish archive properties(
                archive_name = "&fname..zip" archive_path = "&outdir"
                );
        ods package(myout_zip) close;
        /* Delete temporary CSV */
        rc = filename("csvfile", &fcsv);
        rc = fdelete("csvfile");
    run;
%mend export_zip;

/* Export final dataset to ZIP */
%export_zip(stocks_final, /home/nwu/joseparreiras/, stocks);
