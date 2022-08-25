# Natural Language based Financial Forecasting

## Objective
The aim of this thesis is to proof the suitability of natural language based forecasting for
price and volatility development by an experiment. In other words, the hypothesis of this
work is that the development of financial figures as price and volatility can be predicted
with Machine Learning (ML) modelling. Thereby, open source libraries for ML will be accessed in a Python
development environment. The input data is sourced from data science platforms and
providers.

## Central Question and Hypothesis
A first issue of the experiment setting is which financial asset or figure should be chosen
to predict. The models will try to forecast the price development of Standard & Poor’s 500
stock index (SPX) and the corresponding Chicago Board Options Exchange (CBOE) Volatility
Index (VIX). The data used for the forecast will be gathered from the American
Association of Individual Investors (AAII) and files from the United States Securities and
Exchange Commission (SEC). The range of prediction will refer to the frequency of data
which is weekly for AAII sentiment and annually for Form 10-K or rather quarterly for
Form 10-Q filings from the SEC’s Electronic Data Gathering, Analysis, and Retrieval
(EDGAR) database. For the prediction model a long short-term memory (LSTM) recurrent
neural network (RNN) is going to be used. <br>
The working hypothesis proofs by an experiment whether it is possible to forecast the
weekly and quarterly price movement of the SPX and VIX rate with AAII sentiment data
and textual information gained from SEC Form 10-K and 10-Q filings. Hence, the following
research issue derives:

Is it possible to partially predict weekly and quarterly development of SPX price
and VIX rate with LSTM using AAII sentiment data and information gained from
Form 10-K and Form 10-Q filings?

## Market Data
The forecast model will refer to market data as price and volatility. The VIX measures
the market’s expectation of future volatility. It’s based on options of the SPX, considered
the leading indicator of broad U.S. stock market. The VIX is recognized as the world’s
premier gauge on U.S. equity market volatility. It estimates expected volatility by aggregating
the weighted price of SPX puts and calls over a wide range of strike prices. Specifically,
the prices used to calculate VIX values are midpoints of real-time SPX option
bid/ask price quotations. It’s used as a barometer for market uncertainty, providing market
participants and observers with a measurement of constant, 30-day expected volatility of
the broad U.S. stock market. It’s not directly tradable, but the VIX methodology provides
a script for replicating volatility exposure with a portfolio of SPX options. Historical Data
is provided on the CBOE website.

## Analytics and Corporate Disclosures
For the experiment setting, analytics as sentiment and textual information will be used.
Analytics can be seen as derivative data and thus the signals are extracted from a raw
source. However, analytics may be costly, the methodology used in their production may
be biased or opaque, and one will not be the sole consumer (Prado 2018, p. 25).

The AAII Investor Sentiment Survey is a weekly survey of its members which asks if
they are “Bullish”, “Bearish”, or “Neutral” on the stock market over the next six months.
AAII first conducted this survey in 1987 via standard mail. In 2000, the survey was moved 
to AAII’s website. Pre-processed data public available on Quandl also includes the SPX
weekly price.

The Form 10-K and Form 10-Q used for text analysis belong to the group of corporate
disclosures. A 10-K is a comprehensive report, filed annualy by a publicly traded company
about its financial performance and is required by the SEC. The report contains
much more detail than a company’s annual report, which is sent to its shareholders before
an annual meeting to elect company directors. Because of the depth and nature of the
information they contain, 10-Ks are fairly long and tend to be complicated. The SEC requires
companies to publish 10-K forms so investors have fundamental information about companies
so they can make informed investment decisions. This form gives a clearer picture
of everything a company does and what kinds of risks it faces. The 10-K includes five
distinct sections:

* Business provides an overview of the company’s main operations, including its
products and services (i.e., how it makes money).

* Risk factors outline any and all risks the company faces or may face in the future.
The risks are typically listed in order of importance.

* Selected financial data details specific financial information about the company
over the last five years. This section presents more of a near-term view of the
company’s recent performance.

* Management’s discussion and analysis of financial condition and results of operations
gives the company an opportunity to explain its business results from the
previous fiscal year. Also known as MD&A, this section is where the company
can tell its story in own words.

* Financial statements and supplementary data include the company’s audited
financial statements including the income statement, balance sheets, and statement
of cash flows. A letter from the company’s independent auditor certifying
the scope of their review is also included in this section.

A 10-K filing also includes signed letters from the company’s chief executive officer and
chief financial officer. In it, the executives swear under oath that the information included
in the 10-K is accurate. These letters became a requirement after several high-profile
cases involving accounting fraud following the dot-com bust.
Notably, 10-K filings are public information and readily available through a number of
sources. In fact, the vast majority of companies include them in the Investor Relations
section of their website.

Form 10-Q must be submitted to the SEC on a quarterly basis. Unlike the 10-K, the information
in the 10-Q is usually unaudited. The company is only required to file it three
times a year as the 10-K is filed in the forth quarter. There are two parts to a 10-Q filing:

* The first part contains relevant financial information covering the period. This
includes condensed financial statements, management discussion and analysis on
the financial condition of the entity, disclosures regarding market risk, and internal
controls.

* The second part contains all other pertinent information. This includes legal proceedings,
unregistered sales of equity securities, the use of proceeds from the sale
of unregistered sales of equity, and defaults upon senior securities. The company
disclosures any other information – including the use of exhibits – in this section.

The 10-Q provides a window into the financial health of the company. Investors can use
the form to see what changes are taking place within the corporation even before it files
its quarterly earnings. Some areas of interest to investors that are commonly visible in the
10-Q include change of working capital and/or accounts receivables, factors affecting a
company’s inventory, share buybacks, and even any legal risks that a company faces.

## File Summaries
The models use a file containing sentiment counts, file size and other measurements for
all 10-K and 10-Q filings generally stated as 10-X filings from 1994 to 2018. This file
contains a header record with labels and is comma delimited. Each record reports:
1. CIK – the SEC Central Index Key.
2. FILING_DATE – the filing date (YYYYMMDD) for the form.
3. FYE – fiscal-year-end as reported in the filing.
4. FORM_TYPE – the specific from type (e.g., 10-K, 10-K/A, 10-Q405, etc.).
5. FILE_NAME – the local file name for the filing.
6. SIC – the four digit SIC reported in the header of the filing. If this number does
not appear in the header, then the primary web page for all filings from that firm
at EDGAR is parsed in an attempt to identify the SIC number. If all of these methods
fail, an SIC of -99 is assigned.
7. FFInd – the Fama-French 48 industry classification based on the SIC number. All
missing SIC’s are assigned to the miscellaneous category.
8. N_Words – the count of all words, where a word is any token appearing in the
Master Dictionary.
9. N_Unique – the number of words occurring at least once in the document.
10. A sequence of sentiment counts – negative, positive, uncertainty, litigious, weak
modal, moderate modal, strong modal, constraining.
11. N_Negation – a count of cases where the negation occurs within four or fewer
words from a word identified as positive. Negation words are no, none, neither,
never, nobody (see Gunel Totie, 1991, Negation in Speech and Writing). Thus the
net positive words is the positive word count minus the count for Negation. Although
the technique seems reasonable, most important cases of negation are sufficiently
subtle that most algorithms will not pick them up.
12. GrossFileSize – the total number of characters in the original filing.
13. NetFileSize – the total number of characters in the filing after the Stage One Parse.
14. ASCIIEncodedChars – the total number of American Standard Code for Information
Interchange (ASCII) Encoded characters (e.g., &amp;).
15. HTMLChars – the total number of characters attributable to Hypertext Markup
Language (HTML) encoding.
16. XBRLChars – the total number of characters attributable to eXtensible Business
Reporting Language (XBRL) encoding.
17. XMLChars – the total number of characters attributable to Extensible Markup
Language (XML) encoding.
18. N_Tables – number of tables in the filing.
19. N_Exhibits – number of exhibits in the filing.

## Methodology
In total, four models are developed:
1. AAII_LSTM.ipynb
2. AAII_VIX_LSTM.ipynb
3. LM_10X_LSTM.ipynb
4. LM_10X_VIX_LSTM.ipynb

The first two models refer to the AAII sentiment data set available on Quandl. The last
two models use the Loughran and McDonald 10X file summaries. Model 1 and 3 predict
the development of the SPX while model 2 and 4 forecast the VIX development.

## Data Sources
AAII sentiment and SPX weekly price: https://www.quandl.com/data/AAII/AAII_SENTIMENT <br>
File Summaries: https://drive.google.com/file/d/12YQ3bczd3-G94eSpqawbA1hwF0Jzs_jB/view?usp=sharing <br>
VIX historical data: http://www.cboe.com/products/vix-index-volatility/vix-options-and-futures/vix-index/vix-historical-data
