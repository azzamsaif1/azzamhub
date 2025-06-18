const ctx = document.getElementById('marketChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      datasets: [{
        label: 'Customer Interest',
        data: [12, 19, 8, 17, 14, 10, 20],
        borderColor: '#2AE66F',
        backgroundColor: 'rgba(42, 230, 111, 0.1)',
        tension: 0.3,
        fill: true,
        pointRadius: 3,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#ffffff' }
        }
      },
      scales: {
        x: {
          ticks: { color: '#ffffff' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          ticks: { color: '#ffffff' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      }
    }
  });


// body chart
const bg = document.getElementById('bgChart').getContext('2d');
new Chart(bg, {
  type: 'line',
  data: {
    labels: Array.from({ length: 30 }, (_, i) => i + 1),
    datasets: [{
      data: Array.from({ length: 30 }, () => Math.floor(Math.random() * 25 + 5)),
      borderColor: '#2AE66F',
      backgroundColor: 'rgba(42, 230, 111, 0.15)',
      fill: true,
      tension: 0.4,
      pointRadius: 0
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 3000,
      easing: 'easeInOutSine'
    },
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { display: false }
    }
  }
});

// soft move
document.querySelector('#planner .open-btn').addEventListener('click', () => {
    document.getElementById('mainModules').style.display = 'none';
    document.getElementById('plannerSection').style.display = 'block';
});
// 🔁 Show/Hide Password
function togglePassword(inputId, icon) {
  const input = document.getElementById(inputId);
  if (input.type === "password") {
    input.type = "text";
    icon.textContent = "🙈";
  } else {
    input.type = "password";
    icon.textContent = "👁️";
  }
}





// تأكد أن الكود مرفق بعد تحميل DOM
// Ensure the script runs after the page loads
function openPlanner() {
  // أظهر قسم التخطيط فقط
  document.getElementById("plannerSection").style.display = "block";

  // أخفِ عناصر غير ضرورية
  const elementsToHide = [
    "heroSection",
    "planner",     // الكرت نفسه
    "auto",
    "cv",
    "lingo",
    "invest",
  ];

  elementsToHide.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });

  // أخفِ قسم AzzamMarketMind الكامل باستخدام class
  const marketMindSection = document.querySelector(".marketmind-section");
  if (marketMindSection) {
    marketMindSection.style.display = "none";
  }

  // أخفِ المخطط البياني إن أردت
  const marketChart = document.getElementById("marketChart");
  if (marketChart) {
    marketChart.parentElement.style.display = "none";  // أخفي الحاوية
  }
}




let tasks = [];
let timerInterval;

// ✅ Add new task
function addTask() {
    const title = document.getElementById("taskTitle").value.trim();
    const dateTime = document.getElementById("taskDateTime").value;
    const type = document.getElementById("taskType").value;
    const subtype = document.getElementById("taskSubtype").value;

    if (!title || !dateTime || !type) {
        const errorDiv = document.getElementById("taskError");
        errorDiv.innerText = "⚠️ Please fill in all fields!";
        errorDiv.style.display = "block";
        errorDiv.style.color = "#e74c3c";
        setTimeout(() => {
            errorDiv.style.display = "none";
        }, 3000);
        return;
    }

    const task = { title, dateTime, type, subtype };
    tasks.push(task);
    displayTasks();
    document.getElementById("taskTitle").value = "";
    document.getElementById("taskDateTime").value = "";
}

// ✅ Display all tasks
function displayTasks() {
    const taskList = document.getElementById("taskList");
    taskList.innerHTML = "";

    tasks.forEach((task, index) => {
        const taskDiv = document.createElement("div");
        taskDiv.classList.add("task-item");

        taskDiv.innerHTML = `
            <strong>${task.title}</strong>
            (${task.type}${task.subtype ? " – " + task.subtype : ""})
            – ${new Date(task.dateTime).toLocaleString()}
            <button class="start-timer-btn" id="start-btn-${index}" onclick="startTimer(${index})">Start Timer</button>
            <div id="timer-box-${index}" class="pomodoro-box" style="display:none;"></div>
        `;

        taskList.appendChild(taskDiv);
    });
}

// ✅ Initial Timer Prompt
function startTimer(index) {
    clearInterval(timerInterval);

    // Hide the Start Timer button
    const startBtn = document.getElementById(`start-btn-${index}`);
    if (startBtn) startBtn.style.display = "none";

    // Show the timer config box
    const timerBox = document.getElementById(`timer-box-${index}`);
    if (!timerBox) return;

    timerBox.style.display = "block";
    timerBox.innerHTML = `
        <div class="pomodoro-config">
            <p>🎯 Do you want to customize your focus session?</p>
            <button onclick="showConfig(${index})">Yes</button>
            <button onclick="startDefaultTimer(${index})">Start Now</button>
        </div>
    `;
}

// ✅ Show configuration input
function showConfig(index) {
    const task = tasks[index];
    const timerBox = document.getElementById(`timer-box-${index}`);

    const defaultDurations = {
        Study: 50, Work: 90, Health: 30, Sleep: 90, Other: 25
    };
    const defaultBreaks = {
        Study: 10, Work: 15, Health: 5, Sleep: 0, Other: 5
    };

    const duration = defaultDurations[task.type] || 25;
    const breakTime = defaultBreaks[task.type] || 5;

    timerBox.innerHTML = `
        <div class="pomodoro-settings">
            <label>⏱️ Duration (minutes):</label>
            <input type="number" id="duration-${index}" value="${duration}" min="10" max="180" />
            <label>🔁 Sessions:</label>
            <input type="number" id="sessions-${index}" value="1" min="1" max="10" />
            <label>🛌 Break time (minutes):</label>
            <input type="number" id="break-${index}" value="${breakTime}" min="1" max="30" />
            <button onclick="startCustomTimer(${index})">Start Session</button>
        </div>
    `;
}




// ✅ Start with custom input
function startCustomTimer(index) {
    const task = tasks[index];
    const minutes = parseInt(document.getElementById(`duration-${index}`).value);
    runTimer(index, minutes, task);
}

// #function speak
function speakMessage(text) {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US'; // or 'de-DE' إ
    utterance.pitch = 1.1;
    utterance.rate = 1;
    synth.speak(utterance);
}
// ✅ Core Timer Function
let sessionLog = [];  // ✅ Define session history globally
let currentSeconds = 0;
let isPaused = false;

function runTimer(index, minutes, task) {
    clearInterval(timerInterval);
    currentSeconds = 0;
    isPaused = false;

    const totalSeconds = minutes * 60;
    const timerBox = document.getElementById(`timer-box-${index}`);

    timerBox.innerHTML = `
        <div class="pomodoro-header">⏳ ${task.type} – ${task.subtype || "Focus Session"}</div>
        <div class="countdown" id="countdown-${index}" style="text-align:center; font-size:2em;">00:00</div>
        <button id="pause-btn-${index}" onclick="pauseTimer(${index})">⏸️ Pause</button>
        <button id="resume-btn-${index}" onclick="resumeTimer(${index})" style="display:none;">▶️ Resume</button>
        <div class="session-message" id="message-${index}"></div>
    `;

    timerInterval = setInterval(() => {
        if (!isPaused && currentSeconds <= totalSeconds) {
            currentSeconds++;
            document.getElementById(`countdown-${index}`).textContent = formatTime(currentSeconds);

            if (currentSeconds === totalSeconds) {
                clearInterval(timerInterval);

                // ✅ Save session to log
                sessionLog.push({
                    title: task.title,
                    type: task.type,
                    subtype: task.subtype,
                    duration: minutes,
                    timestamp: new Date()
                    
                    
                });
                
                document.getElementById(`message-${index}`).textContent =
                `✅ Session Completed! Great job on your ${task.subtype || task.type} session!`;
            
                // ✅ إعداد الرسالة
                const message =` Session completed. Great job on your ${task.subtype || task.type} session!`;
                document.getElementById(`message-${index}`).textContent =` ✅ ${message}`;

                // ✅ تحويل النص إلى كلام
                const utterance = new SpeechSynthesisUtterance(message);
                utterance.lang = 'en-US';  // أو 'ar-SA' للصوت العربي
                speechSynthesis.speak(utterance);

                // ✅ Display success message
                const messageBox = document.getElementById(`message-${index}`);
                messageBox.innerHTML = `
                    ✅ Session Completed! Great job on your ${task.subtype || task.type} session!<br>
                    <strong>🧠 Summary:</strong><br>
                    You focused for <span style="color:#2AE66F;">${minutes} minutes</span> on <span style="color:#2AE66F;">${task.subtype || task.type}</span>.<br>
                    Total sessions logged: <strong>${sessionLog.length}</strong>.
                `;
            }
        }
    }, 1000);
}

// Pause Timer
function pauseTimer(index) {
    isPaused = true;
      document.getElementById(`pause-btn-${index}`).style.display = "none";
      document.getElementById(`resume-btn-${index}`).style.display = "inline-block";
}

// Resume Timer
function resumeTimer(index) {
    isPaused = false;
    document.getElementById(`pause-btn-${index}`).style.display = "inline-block";
    document.getElementById(`resume-btn-${index}`).style.display = "none";
}
// ##function start Break
function startBreak(index, task) {
    let breakMinutes = 5;  // الافتراضي أو يمكن نأخذ من config
    let breakSeconds = breakMinutes * 60;
    const messageBox = document.getElementById(`message-${index}`);

    messageBox.innerHTML = `
        🛌 Break in progress...<br>
        <div class="countdown" id="break-countdown-${index}" style="font-size: 1.5em; margin: 5px 0;">${formatTime(breakSeconds)}</div>
    `;

    const breakInterval = setInterval(() => {
        breakSeconds--;
        document.getElementById(`break-countdown-${index}`).textContent = formatTime(breakSeconds);

        if (breakSeconds <= 0) {
            clearInterval(breakInterval);

            // ✅ نهاية الباوزا + زر استئناف
            messageBox.innerHTML += `
                ✅ Break completed!<br>
                <button onclick="alert('🎉 Ready for next session?')">Start Next</button>
            `;

            const utter = new SpeechSynthesisUtterance("Break over. Ready for your next focus session?");
            speechSynthesis.speak(utter);
        }
    }, 1000);
}

// Format Time Function
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return ` ${mins < 10 ? "0" : ""}${mins}:${secs < 10 ? "0" : ""}${secs}`;
}// ✅ Return default time based on task type
function getTaskMinutes(type) {
    switch (type) {
        case "Study": return 50;
        case "Work": return 90;
        case "Health": return 30;
        case "Sleep": return 90;
        default: return 25;
    }
}

// ✅ Format countdown time
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
}



// ✅ Automatically update subtype options based on task type
document.getElementById("taskType").addEventListener("change", function () {
    const subtype = document.getElementById("taskSubtype");
    const type = this.value;

    // Clear previous options
    subtype.innerHTML = "";

    // Options for each type
    let options = [];

    if (type === "Study") {
        options = ["Math", "Physics", "Programming", "Chemistry", "Review"];
    } else if (type === "Work") {
        options = ["Coding", "Meetings", "Emails", "Reports"];
    } else if (type === "Health") {
        options = ["Gym", "Yoga", "Running", "Cycling"];
    } else if (type === "Sleep") {
        options = ["Night", "Nap"];
    } else if (type === "Other") {
        options = ["Planning", "Thinking", "Reading"];
    }

    // If no options, hide the select
    if (options.length === 0) {
        subtype.style.display = "none";
    } else {
        // Otherwise show and fill it
        subtype.style.display = "inline-block";
        options.forEach(opt => {
            const option = document.createElement("option");
            option.value = opt;
            option.textContent = opt;
            subtype.appendChild(option);
        });
    }
});

// ###session log in
