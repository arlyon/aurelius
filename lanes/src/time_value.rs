use anyhow::{Result, ensure};
use chrono::{DateTime, FixedOffset, NaiveDate};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TimeBound {
    Date { value: NaiveDate },
    Timestamp { value: DateTime<FixedOffset> },
    Month { year: i32, month: u32 },
}
impl TimeBound {
    pub fn validate(&self) -> Result<()> {
        if let Self::Month { year, month } = self {
            ensure!(
                (1..=9999).contains(year) && (1..=12).contains(month),
                "invalid month"
            );
        }
        Ok(())
    }
    fn literal_matches(&self, text: &str) -> bool {
        match self {
            Self::Date { value } => [
                "%Y-%m-%d",
                "%B %-d, %Y",
                "%b %-d, %Y",
                "%-d %B %Y",
                "%-d %b %Y",
            ]
            .iter()
            .any(|format| text == value.format(format).to_string()),
            Self::Timestamp { value } => DateTime::parse_from_rfc3339(text)
                .is_ok_and(|source| source == *value && source.offset() == value.offset()),
            Self::Month { year, month } => {
                NaiveDate::from_ymd_opt(*year, *month, 1).is_some_and(|date| {
                    ["%Y-%m", "%B %Y", "%b %Y"]
                        .iter()
                        .any(|format| text == date.format(format).to_string())
                })
            }
        }
    }
    fn compare(&self, other: &Self) -> Result<Ordering> {
        self.validate()?;
        other.validate()?;
        match (self, other) {
            (Self::Date { value: a }, Self::Date { value: b }) => Ok(a.cmp(b)),
            (Self::Timestamp { value: a }, Self::Timestamp { value: b }) => Ok(a.cmp(b)),
            (Self::Month { year: a, month: b }, Self::Month { year: c, month: d }) => {
                Ok((a, b).cmp(&(c, d)))
            }
            _ => anyhow::bail!("mixed temporal precision needs additional evidence"),
        }
    }
    pub fn grounded_by(&self, text: &str) -> bool {
        match self {
            Self::Date { value } => [
                "%Y-%m-%d",
                "%B %-d, %Y",
                "%b %-d, %Y",
                "%-d %B %Y",
                "%-d %b %Y",
            ]
            .iter()
            .any(|format| text.contains(&value.format(format).to_string())),
            Self::Timestamp { value } => {
                text.contains(&value.to_rfc3339())
                    || (value.offset().local_minus_utc() == 0
                        && text
                            .contains(&value.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)))
            }
            Self::Month { year, month } => {
                NaiveDate::from_ymd_opt(*year, *month, 1).is_some_and(|date| {
                    ["%Y-%m", "%B %Y", "%b %Y"]
                        .iter()
                        .any(|format| text.contains(&date.format(format).to_string()))
                })
            }
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TimeValue {
    Date {
        value: NaiveDate,
    },
    Timestamp {
        value: DateTime<FixedOffset>,
    },
    Month {
        year: i32,
        month: u32,
    },
    Relative {
        text: String,
    },
    Period {
        start: TimeBound,
        end: TimeBound,
        end_inclusive: bool,
    },
}
impl TimeValue {
    pub fn literal_matches(&self, text: &str) -> bool {
        if let Some(bound) = self.bound() {
            return bound.literal_matches(text);
        }
        match self {
            Self::Relative { text: expression } => text == expression,
            Self::Period { .. } => self.grounded_by(text),
            _ => false,
        }
    }
    fn bound(&self) -> Option<TimeBound> {
        match self {
            Self::Date { value } => Some(TimeBound::Date { value: *value }),
            Self::Timestamp { value } => Some(TimeBound::Timestamp { value: *value }),
            Self::Month { year, month } => Some(TimeBound::Month {
                year: *year,
                month: *month,
            }),
            _ => None,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(bound) = self.bound() {
            return bound.validate();
        }
        match self {
            Self::Relative { text } => ensure!(
                !text.trim().is_empty() && text.len() <= 200,
                "invalid unresolved date expression"
            ),
            Self::Period {
                start,
                end,
                end_inclusive,
            } => ensure!(
                match start.compare(end)? {
                    Ordering::Less => true,
                    Ordering::Equal => *end_inclusive,
                    Ordering::Greater => false,
                },
                "invalid temporal period"
            ),
            _ => (),
        }
        Ok(())
    }
    pub fn grounded_by(&self, text: &str) -> bool {
        if let Some(bound) = self.bound() {
            return bound.grounded_by(text);
        }
        match self {
            Self::Relative { text: expression } => text.contains(expression),
            Self::Period { start, end, .. } => start.grounded_by(text) && end.grounded_by(text),
            _ => false,
        }
    }
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Validity {
    pub from: Option<TimeBound>,
    pub to: Option<TimeBound>,
}
impl Validity {
    pub fn validate(&self) -> Result<()> {
        for value in self.from.iter().chain(self.to.iter()) {
            value.validate()?;
        }
        if let (Some(start), Some(end)) = (&self.from, &self.to) {
            ensure!(
                start.compare(end)? == Ordering::Less,
                "invalid validity interval"
            );
        }
        Ok(())
    }
    pub fn grounded_by(&self, text: &str) -> bool {
        self.from
            .iter()
            .chain(self.to.iter())
            .all(|value| value.grounded_by(text))
    }
}
