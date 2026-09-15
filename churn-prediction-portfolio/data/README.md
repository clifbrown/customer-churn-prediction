# Dataset provenance and scope

`raw/telco_churn.csv` is the unmodified dataset supplied inside the user's original archive. It contains 7,043 rows and 21 columns; the 19 predictors exclude customerID and Churn.

SHA-256:

`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

The archive names [IBM's Telco churn example repository](https://github.com/IBM/telco-customer-churn-on-icp4d) as the source. That repository links its [Telco-Customer-Churn.csv](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) and publishes an [Apache-2.0 repository licence](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE). The uploaded copy's original download history and byte-for-byte identity to an upstream revision have not been independently attested; the supplied bytes are the input to this study. No new dataset licence or ownership claim is introduced here.

IBM describes its [Telco sample](https://www.ibm.com/docs/en/cognos-analytics/12.1.x?topic=samples-telco-customer-churn) as a fictional-company dataset. This project demonstrates genuine computation on that educational sample, not analysis of independently verified real commercial customer events.

The file has no dated feature snapshots. Its label prevalence must not be described as annual industry churn, and the model must not be described as a validated forecast of next-month churn. A live use case needs a defined prediction date, horizon, and feature-availability audit.

`sample_customers.csv` contains ten profiles from the training partition with the target removed. It is a convenience input for the dashboard and CLI, not a separate evaluation dataset.
