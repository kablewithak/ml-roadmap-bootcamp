#!/usr/bin/env python3
"""
Code Mastery CLI - Your Daily Learning Companion
Run this every day to track progress and stay motivated
"""

import os
import json
import datetime
import random
import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import textwrap

class Colors:
    """Terminal colors for better UX"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CodeMasteryCLI:
    """Daily CLI for tracking 52,000 lines journey"""
    
    def __init__(self):
        self.data_file = Path("mastery_progress.json")
        self.target_date = datetime.datetime(2025, 1, 31)
        self.target_lines = 52000
        self.load_data()
        
        # Motivational quotes for daily inspiration
        self.quotes = [
            "Code is like humor. When you have to explain it, it's bad. - Cory House",
            "First, solve the problem. Then, write the code. - John Johnson",
            "The best error message is the one that never shows up. - Thomas Fuchs",
            "Simplicity is the soul of efficiency. - Austin Freeman",
            "Make it work, make it right, make it fast. - Kent Beck",
            "Every line of code is a liability. - Tef",
            "The best code is no code at all. - Jeff Atwood",
            "Debugging is twice as hard as writing code. - Brian Kernighan",
            "Code never lies, comments sometimes do. - Ron Jeffries",
            "Perfection is achieved when there is nothing left to take away. - Antoine de Saint-Exupéry"
        ]
        
        # Pattern achievement badges
        self.badges = {
            100: "🎯 First Hundred!",
            500: "🔥 Demon Slayer!",
            1000: "⚡ Kilocode Warrior!",
            5000: "🚀 Pattern Master!",
            10000: "💎 Architecture Sage!",
            20000: "🏆 System Designer!",
            30000: "👑 Code Sovereign!",
            40000: "🌟 Engineering Lord!",
            50000: "🎖️ LEGENDARY MASTER!",
            52000: "🏅 MISSION COMPLETE!"
        }
    
    def load_data(self):
        """Load or initialize progress data"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
            # This is a "Schema Migration" on the fly
            if "achievements" not in self.data:
                self.data["achievements"] = []
        else:
            self.data = {
                "total_lines": 0,
                "total_patterns": 0,
                "daily_history": [],
                "patterns_learned": [],
                "achievements": [],
                "start_date": datetime.datetime.now().isoformat()
            }
            self.save_data()
    
    def save_data(self):
        """Save progress data"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_progress(self, lines: int, patterns: List[str], insights: str):
        """Add today's progress"""
        today = datetime.datetime.now()
        
        # Update totals
        self.data["total_lines"] += lines
        self.data["total_patterns"] += len(patterns)
        
        # Add unique patterns
        for pattern in patterns:
            if pattern and pattern not in self.data["patterns_learned"]:
                self.data["patterns_learned"].append(pattern)
        
        # Add daily entry
        entry = {
            "date": today.isoformat(),
            "lines": lines,
            "patterns": patterns,
            "insights": insights,
            "total_so_far": self.data["total_lines"]
        }
        self.data["daily_history"].append(entry)
        
        # Check for new achievements
        self.check_achievements()
        
        self.save_data()
    
    def check_achievements(self):
        """Check and award achievement badges"""
        total = self.data["total_lines"]
        for threshold, badge in self.badges.items():
            if total >= threshold and badge not in self.data["achievements"]:
                self.data["achievements"].append(badge)
                print(f"\n{Colors.YELLOW}🎊 NEW ACHIEVEMENT UNLOCKED! 🎊{Colors.ENDC}")
                print(f"{Colors.BOLD}{badge}{Colors.ENDC}\n")
    
    def show_dashboard(self):
        """Display comprehensive progress dashboard"""
        now = datetime.datetime.now()
        days_elapsed = (now - datetime.datetime.fromisoformat(self.data["start_date"])).days + 1
        days_remaining = (self.target_date - now).days
        lines_remaining = self.target_lines - self.data["total_lines"]
        
        # Calculate rates
        current_rate = self.data["total_lines"] / max(days_elapsed, 1)
        required_rate = lines_remaining / max(days_remaining, 1)
        
        # Progress bar
        progress_pct = min(self.data["total_lines"] / self.target_lines * 100, 100)
        bar_length = 50
        filled = int(bar_length * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Clear screen for better display
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Header
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}        CODE MASTERY DASHBOARD - ROAD TO 52,000 LINES{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        # Progress Bar
        print(f"{Colors.BOLD}Overall Progress:{Colors.ENDC}")
        print(f"[{bar}] {progress_pct:.1f}%")
        print(f"{self.data['total_lines']:,} / {self.target_lines:,} lines\n")
        
        # Statistics
        print(f"{Colors.BOLD}📊 Statistics:{Colors.ENDC}")
        print(f"  Days Elapsed:    {days_elapsed}")
        print(f"  Days Remaining:  {days_remaining}")
        print(f"  Current Pace:    {current_rate:.0f} lines/day")
        print(f"  Required Pace:   {required_rate:.0f} lines/day")
        
        if current_rate >= required_rate:
            status = f"{Colors.GREEN}✅ ON TRACK{Colors.ENDC}"
        elif current_rate >= required_rate * 0.8:
            status = f"{Colors.YELLOW}⚠️  SLIGHT ADJUSTMENT NEEDED{Colors.ENDC}"
        else:
            status = f"{Colors.RED}🚨 ACCELERATION REQUIRED{Colors.ENDC}"
        
        print(f"  Status:          {status}\n")
        
        # Patterns & Achievements
        print(f"{Colors.BOLD}🧩 Patterns Mastered:{Colors.ENDC} {len(self.data['patterns_learned'])}")
        if self.data['patterns_learned']:
            recent_patterns = self.data['patterns_learned'][-5:]
            for pattern in recent_patterns:
                print(f"  • {pattern}")
        
        print(f"\n{Colors.BOLD}🏆 Achievements:{Colors.ENDC} {len(self.data['achievements'])}")
        if self.data['achievements']:
            for achievement in self.data['achievements'][-3:]:
                print(f"  {achievement}")
        
        # Recent History
        print(f"\n{Colors.BOLD}📈 Last 7 Days:{Colors.ENDC}")
        recent = self.data['daily_history'][-7:]
        for entry in recent:
            date = datetime.datetime.fromisoformat(entry['date']).strftime("%b %d")
            lines = entry['lines']
            symbol = "🟢" if lines >= 750 else "🟡" if lines >= 500 else "🔴"
            print(f"  {date}: {symbol} {lines} lines")
        
        # Daily Quote
        quote = random.choice(self.quotes)
        print(f"\n{Colors.CYAN}💭 Daily Inspiration:{Colors.ENDC}")
        wrapped = textwrap.wrap(f'"{quote}"', width=65)
        for line in wrapped:
            print(f"  {line}")
        
        # Footer
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        
        # Recommendations
        self.show_recommendations(current_rate, required_rate)
    
    def show_recommendations(self, current_rate: float, required_rate: float):
        """Show personalized recommendations"""
        print(f"\n{Colors.BOLD}💡 Today's Recommendations:{Colors.ENDC}")
        
        if current_rate < required_rate:
            deficit = required_rate - current_rate
            print(f"  • Increase daily time by {deficit/50:.0f} minutes")
            print(f"  • Focus on high-value components (payment, streaming)")
        
        if len(self.data['patterns_learned']) < 10:
            print(f"  • Focus on pattern recognition - aim for 2 patterns/day")
        
        hour = datetime.datetime.now().hour
        if hour < 8:
            print(f"  • Perfect time for deep focus! Start with core systems")
        elif hour < 12:
            print(f"  • Great morning energy - tackle complex patterns")
        elif hour < 17:
            print(f"  • Afternoon: Good for integration and connections")
        else:
            print(f"  • Evening: Perfect for synthesis and documentation")
        
        # Streak tracking
        if len(self.data['daily_history']) > 0:
            last_date = datetime.datetime.fromisoformat(self.data['daily_history'][-1]['date']).date()
            today = datetime.datetime.now().date()
            if last_date == today - datetime.timedelta(days=1):
                print(f"  • 🔥 Keep your streak alive! Log today's progress")
    
    def quick_log(self):
        """Quick logging interface"""
        print(f"\n{Colors.BOLD}📝 Quick Progress Log{Colors.ENDC}")
        print("-" * 40)
        
        # Get lines studied
        while True:
            try:
                lines = int(input(f"{Colors.CYAN}Lines studied today: {Colors.ENDC}"))
                break
            except ValueError:
                print(f"{Colors.RED}Please enter a number{Colors.ENDC}")
        
        # Get patterns
        patterns_input = input(f"{Colors.CYAN}Patterns identified (comma-separated): {Colors.ENDC}")
        patterns = [p.strip() for p in patterns_input.split(',') if p.strip()]
        
        # Get key insight
        insight = input(f"{Colors.CYAN}Key insight or learning: {Colors.ENDC}")
        
        # Add progress
        self.add_progress(lines, patterns, insight)
        
        # Show summary
        print(f"\n{Colors.GREEN}✅ Progress logged successfully!{Colors.ENDC}")
        print(f"Total Progress: {self.data['total_lines']:,} / {self.target_lines:,} lines")
        
        if lines >= 750:
            print(f"{Colors.GREEN}🎯 Daily target achieved! Great job!{Colors.ENDC}")
        else:
            remaining = 750 - lines
            print(f"{Colors.YELLOW}📚 Study {remaining} more lines to hit daily target{Colors.ENDC}")
    
    def show_patterns(self):
        """Display pattern library"""
        print(f"\n{Colors.BOLD}🧩 PATTERN LIBRARY{Colors.ENDC}")
        print("=" * 50)
        
        if not self.data['patterns_learned']:
            print("No patterns learned yet. Start identifying patterns in your code!")
            return
        
        # Group patterns by category
        categories = {
            "Distributed": ["lock", "saga", "outbox", "consensus"],
            "Resilience": ["circuit", "retry", "bulkhead", "timeout"],
            "Data": ["event", "cqrs", "sourcing", "streaming"],
            "Architecture": ["multi-tenant", "microservice", "gateway"],
            "ML": ["feature", "shadow", "ab_test", "online_learning"]
        }
        
        categorized = {cat: [] for cat in categories}
        uncategorized = []
        
        for pattern in self.data['patterns_learned']:
            found = False
            for cat, keywords in categories.items():
                if any(kw in pattern.lower() for kw in keywords):
                    categorized[cat].append(pattern)
                    found = True
                    break
            if not found:
                uncategorized.append(pattern)
        
        # Display categorized patterns
        for cat, patterns in categorized.items():
            if patterns:
                print(f"\n{Colors.BOLD}{cat} Patterns:{Colors.ENDC}")
                for p in patterns:
                    print(f"  • {p}")
        
        if uncategorized:
            print(f"\n{Colors.BOLD}Other Patterns:{Colors.ENDC}")
            for p in uncategorized:
                print(f"  • {p}")
        
        print(f"\n{Colors.CYAN}Total Patterns Mastered: {len(self.data['patterns_learned'])}{Colors.ENDC}")
    
    def export_report(self):
        """Export detailed progress report"""
        filename = f"progress_report_{datetime.datetime.now().strftime('%Y%m%d')}.md"
        
        with open(filename, 'w') as f:
            f.write("# Code Mastery Progress Report\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Total Lines: {self.data['total_lines']:,} / {self.target_lines:,}\n")
            f.write(f"- Progress: {self.data['total_lines']/self.target_lines*100:.1f}%\n")
            f.write(f"- Patterns Learned: {len(self.data['patterns_learned'])}\n")
            f.write(f"- Achievements: {len(self.data['achievements'])}\n\n")
            
            f.write("## Patterns Mastered\n")
            for pattern in self.data['patterns_learned']:
                f.write(f"- {pattern}\n")
            
            f.write("\n## Daily Progress\n")
            f.write("| Date | Lines | Total | Patterns |\n")
            f.write("|------|-------|-------|----------|\n")
            
            for entry in self.data['daily_history'][-30:]:  # Last 30 days
                date = datetime.datetime.fromisoformat(entry['date']).strftime('%Y-%m-%d')
                f.write(f"| {date} | {entry['lines']} | {entry['total_so_far']} | {len(entry['patterns'])} |\n")
        
        print(f"{Colors.GREEN}✅ Report exported to {filename}{Colors.ENDC}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Code Mastery CLI - Track your journey to 52,000 lines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mastery.py             # Show dashboard
  python mastery.py log          # Quick log progress
  python mastery.py patterns     # View pattern library
  python mastery.py export       # Export detailed report
  python mastery.py motivate     # Get motivation boost
        """
    )
    
    parser.add_argument('command', nargs='?', default='dashboard',
                       choices=['dashboard', 'log', 'patterns', 'export', 'motivate'],
                       help='Command to execute')
    
    args = parser.parse_args()
    cli = CodeMasteryCLI()
    
    if args.command == 'dashboard':
        cli.show_dashboard()
    elif args.command == 'log':
        cli.quick_log()
    elif args.command == 'patterns':
        cli.show_patterns()
    elif args.command == 'export':
        cli.export_report()
    elif args.command == 'motivate':
        # Special motivation mode
        quotes = [
            "You're building the future, one line at a time!",
            "Every pattern you learn makes you unstoppable!",
            "52,000 lines seems far until it's done!",
            "You're not just learning code, you're mastering systems!",
            "Today's confusion is tomorrow's expertise!",
            "The code you understand today saves millions tomorrow!",
            "You're closer to mastery than you were yesterday!",
            "Every bug you understand is a crisis prevented!",
            "Your future self will thank you for today's effort!",
            "Champions are made in the daily grind!"
        ]
        
        print(f"\n{Colors.BOLD}{Colors.YELLOW}⚡ MOTIVATION BOOST ⚡{Colors.ENDC}")
        print("=" * 50)
        selected = random.sample(quotes, 3)
        for quote in selected:
            wrapped = textwrap.wrap(quote, width=45)
            for line in wrapped:
                print(f"  {Colors.CYAN}{line}{Colors.ENDC}")
            print()
        
        print(f"{Colors.GREEN}Now go crush those 750 lines! 💪{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
