"""Training-time feature preparation."""
import pandas as pd


def add_prior_appointments(df, patient_col="PatientID", date_col="Date", id_col="AppointmentID"):
    """Add PriorAppointmentsCount: the number of the patient's appointments strictly
    before this one (ordered by date, then appointment ID for same-day visits).

    Replaces Week 6's PreviousAppointmentsCount (total visits - 1), which counted
    future visits a live system could not know at prediction time.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    order = out.sort_values([patient_col, date_col, id_col]).index
    out.loc[order, "PriorAppointmentsCount"] = out.loc[order].groupby(patient_col).cumcount().values
    out["PriorAppointmentsCount"] = out["PriorAppointmentsCount"].astype("int64")
    return out
