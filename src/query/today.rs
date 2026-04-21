use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PulseEvent {
    HighEq {
        subject: String,
        predicate: String,
        object: String,
        days_until: i64,
    },
    Neuron {
        subject: String,
        predicate: String,
        object: String,
        confidence: f32,
    },
    Health {
        total_facts: usize,
        decayed_count: usize,
        neurons_merged: usize,
        contradictions_resolved: usize,
    },
    BlockedAction {
        action: String,
        client: String,
        invoice_id: String,
    },
}

struct Pulse {
    high_eq: Vec<(String, String, String, i64)>,
    neurons: Vec<(String, String, String, f32)>,
    health: Option<(usize, usize, usize, usize)>,
    blocked: Vec<(String, String, String)>,
}

fn load_pulse(path: impl AsRef<std::path::Path>) -> Result<Pulse> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut pulse = Pulse {
        high_eq: Vec::new(),
        neurons: Vec::new(),
        health: None,
        blocked: Vec::new(),
    };
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<PulseEvent>(&line) {
            Ok(PulseEvent::HighEq {
                subject,
                predicate,
                object,
                days_until,
            }) => {
                pulse.high_eq.push((subject, predicate, object, days_until));
            }
            Ok(PulseEvent::Neuron {
                subject,
                predicate,
                object,
                confidence,
            }) => {
                pulse.neurons.push((subject, predicate, object, confidence));
            }
            Ok(PulseEvent::Health {
                total_facts,
                decayed_count,
                neurons_merged,
                contradictions_resolved,
            }) => {
                pulse.health = Some((
                    total_facts,
                    decayed_count,
                    neurons_merged,
                    contradictions_resolved,
                ));
            }
            Ok(PulseEvent::BlockedAction {
                action,
                client,
                invoice_id,
            }) => {
                pulse.blocked.push((action, client, invoice_id));
            }
            Err(_) => {}
        }
    }
    Ok(pulse)
}

/// Render a confidence bar like `████████░░ 80%` into a string.
fn conf_bar(conf: f32, width: usize) -> String {
    let filled = ((conf * width as f32).round() as usize).min(width);
    let empty = width - filled;
    format!(
        "{}{} {:.0}%",
        "█".repeat(filled),
        "░".repeat(empty),
        conf * 100.0
    )
}

fn render_left(pulse: &Pulse, visible: bool) -> List<'static> {
    let block = Block::default()
        .title(" High-EQ Events ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));

    if !visible {
        return List::new(Vec::<ListItem>::new()).block(block);
    }

    let items: Vec<ListItem> = if pulse.high_eq.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "  No upcoming events",
            Style::default().fg(Color::DarkGray),
        )))]
    } else {
        pulse
            .high_eq
            .iter()
            .map(|(subject, predicate, object, days)| {
                let label = match predicate.as_str() {
                    "born_on" => "birthday",
                    "life_event" => "event",
                    other => other,
                };
                let days_str = match *days {
                    0 => "today!".to_string(),
                    1 => "tomorrow".to_string(),
                    d => format!("in {d}d"),
                };
                let color = if *days <= 1 {
                    Color::Yellow
                } else if *days <= 3 {
                    Color::Green
                } else {
                    Color::White
                };
                let bold = if *days <= 1 {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("  {subject}"),
                        Style::default().fg(color).add_modifier(bold),
                    ),
                    Span::styled(format!(" [{label}]"), Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!(" — {object} {days_str}"),
                        Style::default().fg(color),
                    ),
                ]))
            })
            .collect()
    };

    List::new(items).block(block)
}

fn render_neurons(pulse: &Pulse, visible: bool) -> List<'static> {
    let block = Block::default()
        .title(" New Neurons ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Magenta));

    if !visible {
        return List::new(Vec::<ListItem>::new()).block(block);
    }

    let items: Vec<ListItem> = if pulse.neurons.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "  No new neurons",
            Style::default().fg(Color::DarkGray),
        )))]
    } else {
        pulse
            .neurons
            .iter()
            .map(|(subject, predicate, object, conf)| {
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("  {subject}"),
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!(" {predicate} "),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(object.clone(), Style::default().fg(Color::White)),
                    ]),
                    Line::from(Span::styled(
                        format!("  {}", conf_bar(*conf, 10)),
                        Style::default().fg(Color::Magenta),
                    )),
                ])
            })
            .collect()
    };

    List::new(items).block(block)
}

fn render_health(pulse: &Pulse, visible: bool) -> Paragraph<'static> {
    let block = Block::default()
        .title(" System Health ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Green));

    if !visible {
        return Paragraph::new(Vec::<Line>::new()).block(block);
    }

    let mut lines: Vec<Line> = Vec::new();

    if let Some((total, decayed, neurons, contras)) = pulse.health {
        lines.push(Line::from(vec![
            Span::styled("  Facts: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                total.to_string(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   Decayed: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                decayed.to_string(),
                Style::default().fg(if decayed > 0 {
                    Color::Yellow
                } else {
                    Color::White
                }),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  Neurons merged: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                neurons.to_string(),
                Style::default().fg(if neurons > 0 {
                    Color::Magenta
                } else {
                    Color::White
                }),
            ),
            Span::styled("   Contradictions: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                contras.to_string(),
                Style::default().fg(if contras > 0 {
                    Color::Red
                } else {
                    Color::White
                }),
            ),
        ]));
    } else {
        lines.push(Line::from(Span::styled(
            "  No health data",
            Style::default().fg(Color::DarkGray),
        )));
    }

    if !pulse.blocked.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Blocked Actions:",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
        for (action, client, invoice_id) in &pulse.blocked {
            lines.push(Line::from(Span::styled(
                format!("    {action} — {client} ({invoice_id})"),
                Style::default().fg(Color::Red),
            )));
        }
    }

    Paragraph::new(lines).block(block)
}

pub async fn run_today() -> Result<()> {
    let path = crate::persistence::morning_pulse_path();
    if !path.exists() {
        println!("No morning pulse found. Run `aurelius dream` first.");
        return Ok(());
    }

    let pulse = load_pulse(&path)?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let start = Instant::now();
    // Panels animate in: left at 100ms, neurons at 200ms, health at 300ms
    const PANEL_DELAY: u64 = 100;

    loop {
        let elapsed = start.elapsed().as_millis() as u64;
        let show_left = elapsed >= PANEL_DELAY;
        let show_neurons = elapsed >= PANEL_DELAY * 2;
        let show_health = elapsed >= PANEL_DELAY * 3;

        terminal.draw(|frame| {
            let size = frame.area();

            // Title bar
            let root = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Min(0),
                    Constraint::Length(1),
                ])
                .split(size);

            let title = Paragraph::new(Line::from(vec![
                Span::styled(
                    " Aurelius ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("Morning Pulse", Style::default().fg(Color::White)),
            ]));
            frame.render_widget(title, root[0]);

            // Main area: left 40% | right 60%
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                .split(root[1]);

            frame.render_widget(render_left(&pulse, show_left), columns[0]);

            // Right column: neurons top, health bottom
            let right_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(columns[1]);

            frame.render_widget(render_neurons(&pulse, show_neurons), right_rows[0]);
            frame.render_widget(render_health(&pulse, show_health), right_rows[1]);

            // Footer
            let footer = Paragraph::new(Line::from(Span::styled(
                " Press q or Esc to exit",
                Style::default().fg(Color::DarkGray),
            )));
            frame.render_widget(footer, root[2]);
        })?;

        let all_visible = show_left && show_neurons && show_health;
        let poll_timeout = if all_visible {
            Duration::from_secs(60)
        } else {
            Duration::from_millis(16)
        };

        if event::poll(poll_timeout)? {
            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
