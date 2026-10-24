# Feature Dictionary — Engineered Features

- **AgeGroup**: Groups patients into clinically meaningful age bands; admission/risk patterns differ sharply by age band rather than linearly by age.
- **HasInsurance**: Insurance status is a known driver of care-seeking behavior and billing/claim outcomes.
- **SeverityScore**: Numeric encoding of diagnosis severity so models can use it directly and so a 'severity trend' can be computed.
- **RevenueCategory**: Coarse billing tier useful for segmentation and as an interpretable feature alongside the raw amount.
- **VisitMonth / VisitQuarter**: Captures seasonality in visit volume and case mix.
- **PreviousAppointmentsCount**: Prior utilization is one of the strongest predictors of future utilization and risk.
- **PriorAdmissionsCount**: Patients with a history of admission are measurably more likely to be admitted again.
- **PriorLabTestsCount / PriorPrescriptionsCount**: Proxy for chronic/complex care burden.
- **PatientAvgPriorSatisfaction**: Prior satisfaction is a leading indicator of future satisfaction (used only as history, not the current visit's own rating).
- **FrequentVisitor**: Binary flag for patients with repeat visits in the observed window.
- **DepartmentWorkload / DoctorWorkload / FacilityUtilizationCount**: Operational load proxies used for staffing and capacity-planning features.
- **ClinicalRiskScore / HighRiskPatient**: Constructed composite risk indicator (age, chronic condition, severity, admission history) used as the Model 4 target since no ground-truth risk label exists in the source data.
- **LengthOfStayCategory**: Categorical banding of length of stay for bed-management-oriented reporting.
