/**
 * Umaid Haveli CRM & AI Agent Command Center
 * Real-Time Two-Way Sync Frontend Application
 */

const API_BASE = '/api';
const SYNC_INTERVAL_MS = 2500;

// Application State
const appState = {
    activeTab: 'ordersTab',
    orderFilter: 'all',
    tableSectionFilter: 'all',
    inventorySearchQuery: '',
    sessionId: 'web_crm_' + Math.random().toString(36).substring(2, 8),
    customerPhone: '9660888489',
    customer: null,
    isChatting: false,
    lastKpi: null,
    orders: [],
    tables: [],
    inventory: []
};

// DOM Elements
const dom = {
    // Top Bar
    syncBadge: document.getElementById('syncStatusBadge'),
    liveClock: document.getElementById('liveClock'),
    btnRefresh: document.getElementById('btnRefresh'),

    // KPI Cards
    kpiActiveOrders: document.getElementById('kpiActiveOrders'),
    kpiTableAvailability: document.getElementById('kpiTableAvailability'),
    kpiOccupancyRate: document.getElementById('kpiOccupancyRate'),
    kpiRevenue: document.getElementById('kpiRevenue'),
    kpiLowStock: document.getElementById('kpiLowStock'),
    badgeOrderCount: document.getElementById('badgeOrderCount'),

    // Tab Buttons & Panels
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanels: document.querySelectorAll('.tab-panel'),

    // Orders Tab
    ordersGrid: document.getElementById('ordersGrid'),
    orderFilterBtns: document.querySelectorAll('#ordersTab .filter-btn'),

    // Tables Tab
    tablesGrid: document.getElementById('tablesGrid'),
    tableFilterBtns: document.querySelectorAll('#tablesTab .filter-btn'),

    // Inventory Tab
    inventorySearch: document.getElementById('inventorySearch'),
    inventoryTableBody: document.getElementById('inventoryTableBody'),

    // Chat Tab & Telemetry
    customerPhoneInput: document.getElementById('customerPhoneInput'),
    btnCustomerLogin: document.getElementById('btnCustomerLogin'),
    customerProfileBadge: document.getElementById('customerProfileBadge'),
    chatSessionBadge: document.getElementById('chatSessionBadge'),
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    btnSendMessage: document.getElementById('btnSendMessage'),
    quickChips: document.querySelectorAll('.quick-chips .chip'),
    telIntent: document.getElementById('telIntent'),
    telPromptTok: document.getElementById('telPromptTok'),
    telCompTok: document.getElementById('telCompTok'),
    telTotalTok: document.getElementById('telTotalTok'),
    telActiveSlots: document.getElementById('telActiveSlots'),
    telExecutedSql: document.getElementById('telExecutedSql'),

    // Toast Container
    toastContainer: document.getElementById('toastContainer')
};

/* ==========================================================================
   INITIALIZATION & CLOCK
   ========================================================================== */
function initApp() {
    startClock();
    setupEventListeners();
    loginCustomer(appState.customerPhone);
    fetchAllData();
    
    // Auto 2-Way Sync Loop (2.5 seconds)
    setInterval(() => {
        fetchAllData(true);
    }, SYNC_INTERVAL_MS);
}

function startClock() {
    function update() {
        const now = new Date();
        dom.liveClock.textContent = now.toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        });
    }
    update();
    setInterval(update, 1000);
}

/* ==========================================================================
   EVENT LISTENERS SETUP
   ========================================================================== */
function setupEventListeners() {
    // Manual Refresh Button
    dom.btnRefresh.addEventListener('click', () => {
        dom.btnRefresh.style.transform = 'rotate(360deg)';
        dom.btnRefresh.style.transition = 'transform 0.5s ease';
        fetchAllData(false);
        showToast('Live CRM data synced!', 'info');
        setTimeout(() => { dom.btnRefresh.style.transform = 'none'; }, 500);
    });

    // Dashboard Tabs Navigation
    dom.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    // Order Status Filters
    dom.orderFilterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            dom.orderFilterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            appState.orderFilter = btn.getAttribute('data-status');
            renderOrders();
        });
    });

    // Table Section Filters
    dom.tableFilterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            dom.tableFilterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            appState.tableSectionFilter = btn.getAttribute('data-section');
            renderTables();
        });
    });

    // Inventory Search Filter
    dom.inventorySearch.addEventListener('input', (e) => {
        appState.inventorySearchQuery = e.target.value.toLowerCase().trim();
        renderInventory();
    });

    // Customer Login & Switch
    if (dom.btnCustomerLogin) {
        dom.btnCustomerLogin.addEventListener('click', () => {
            const phone = dom.customerPhoneInput.value.trim();
            if (phone) {
                loginCustomer(phone);
            }
        });
    }

    // Chat Actions
    dom.btnSendMessage.addEventListener('click', handleSendMessage);
    dom.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Chat Quick Chips
    document.querySelectorAll('.quick-chips .chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            dom.chatInput.value = prompt;
            handleSendMessage();
        });
    });

    // Bill Modal Close buttons
    const btnBillModalClose = document.getElementById('btnBillModalClose');
    const btnBillModalDismiss = document.getElementById('btnBillModalDismiss');
    if (btnBillModalClose) btnBillModalClose.addEventListener('click', closeBillModal);
    if (btnBillModalDismiss) btnBillModalDismiss.addEventListener('click', closeBillModal);
}

function switchTab(tabId) {
    appState.activeTab = tabId;
    dom.tabBtns.forEach(b => {
        if (b.getAttribute('data-tab') === tabId) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });

    dom.tabPanels.forEach(panel => {
        if (panel.id === tabId) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    });
}

/* ==========================================================================
   DATA FETCHING & 2-WAY SYNC
   ========================================================================== */
async function fetchAllData(isBackground = false) {
    try {
        const [kpiRes, ordersRes, tablesRes, inventoryRes] = await Promise.all([
            fetch(`${API_BASE}/kpi`).then(r => r.json()),
            fetch(`${API_BASE}/orders`).then(r => r.json()),
            fetch(`${API_BASE}/tables`).then(r => r.json()),
            fetch(`${API_BASE}/inventory`).then(r => r.json())
        ]);

        if (kpiRes.success && kpiRes.data) updateKPIs(kpiRes.data);
        if (ordersRes.success && ordersRes.orders) {
            appState.orders = ordersRes.orders;
            renderOrders();
        }
        if (tablesRes.success && tablesRes.tables) {
            appState.tables = tablesRes.tables;
            renderTables();
        }
        if (inventoryRes.success && inventoryRes.inventory) {
            appState.inventory = inventoryRes.inventory;
            renderInventory();
        }

        // Pulse sync badge
        setSyncStatus(true);
    } catch (err) {
        console.error('Data sync failed:', err);
        setSyncStatus(false);
    }
}

function setSyncStatus(isLive) {
    if (isLive) {
        dom.syncBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        dom.syncBadge.querySelector('.sync-text').textContent = '2-Way DB Sync: Live';
        dom.syncBadge.querySelector('.pulse-dot').style.background = '#10b981';
    } else {
        dom.syncBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        dom.syncBadge.querySelector('.sync-text').textContent = 'Sync Offline';
        dom.syncBadge.querySelector('.pulse-dot').style.background = '#ef4444';
    }
}

function updateKPIs(kpi) {
    appState.lastKpi = kpi;
    const active = kpi.active_orders ?? 0;
    const totalTables = kpi.total_tables ?? 0;
    const availTables = kpi.available_tables ?? 0;
    const occRate = totalTables > 0 ? Math.round(((totalTables - availTables) / totalTables) * 100) : 0;

    dom.kpiActiveOrders.textContent = active;
    dom.badgeOrderCount.textContent = active;
    dom.kpiTableAvailability.textContent = `${availTables} / ${totalTables}`;
    dom.kpiOccupancyRate.textContent = `${occRate}% Occupied`;
    dom.kpiRevenue.textContent = `₹${(kpi.total_revenue || 0).toLocaleString('en-IN')}`;
    dom.kpiLowStock.textContent = kpi.low_stock_count ?? 0;
}

/* ==========================================================================
   RENDER: LIVE KITCHEN ORDERS
   ========================================================================== */
function renderOrders() {
    let filtered = appState.orders;
    if (appState.orderFilter !== 'all') {
        const filterVal = appState.orderFilter.toLowerCase();
        filtered = appState.orders.filter(o => {
            const st = (o.status || '').toLowerCase();
            if (filterVal === 'in_kitchen' || filterVal === 'cooking') {
                return st === 'cooking' || st === 'in_kitchen' || st === 'preparing';
            }
            return st === filterVal;
        });
    }

    if (!filtered || filtered.length === 0) {
        dom.ordersGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🍳</div>
                <h3>No Orders Found</h3>
                <p>No live orders matching filter "${appState.orderFilter}". Place a new order via AI Assistant or CLI.</p>
            </div>
        `;
        return;
    }

    dom.ordersGrid.innerHTML = filtered.map(order => {
        const itemsHtml = (order.items || []).map(item => `
            <div class="order-item-row">
                <span class="item-name">${item.item_name || 'Dish'} × ${item.quantity}</span>
                <span class="item-price">₹${item.total_price}</span>
            </div>
        `).join('');

        const statusClass = (order.status || 'cooking').toLowerCase();
        const formattedTime = order.created_at ? new Date(order.created_at).toLocaleTimeString('en-IN', {
            hour: '2-digit', minute: '2-digit'
        }) : 'Just now';

        let actionButtonsHtml = '';
        if (statusClass === 'completed') {
            actionButtonsHtml = '<div class="order-final-status-badge">🔒 Completed & Settled</div>';
        } else if (statusClass === 'cancelled') {
            actionButtonsHtml = '<div class="order-final-status-badge cancelled">❌ Cancelled</div>';
        } else if (statusClass === 'served') {
            actionButtonsHtml = `
                <button class="action-btn btn-completed" style="grid-column: span 2;" onclick="updateOrderStatus(${order.id}, 'completed')">💳 Settle & Complete</button>
            `;
        } else {
            // cooking / pending / in_kitchen
            actionButtonsHtml = `
                <button class="action-btn btn-served" onclick="updateOrderStatus(${order.id}, 'served')">🍽️ Mark Served</button>
                <button class="action-btn btn-cancelled" onclick="updateOrderStatus(${order.id}, 'cancelled')">❌ Cancel Order</button>
            `;
        }

        return `
            <div class="order-card status-${statusClass}" data-order-id="${order.id}">
                <div class="order-card-header">
                    <div>
                        <span class="order-id">#${order.order_number || ('ORD-' + order.id)}</span>
                        <h4 class="order-table-name">Table ${order.table_number || 'T-??'} (${order.section || 'Main'})</h4>
                    </div>
                    <div class="order-badge status-${statusClass}">${statusClass.toUpperCase()}</div>
                </div>

                <div class="order-card-body">
                    <div class="order-meta-info">
                        <span>🏷️ ${order.order_type || 'Dine-In'}</span>
                        <span>🕒 ${formattedTime}</span>
                        <span>💳 ${order.status === 'completed' ? 'Paid' : 'Pending'}</span>
                    </div>

                    <div class="order-items-list">
                        ${itemsHtml || '<span style="color:var(--text-muted); font-size:0.85rem;">No items listed</span>'}
                    </div>

                    <div class="order-total-row">
                        <span>Net Amount</span>
                        <strong>₹${(order.net_amount || order.total_amount || 0).toLocaleString('en-IN')}</strong>
                    </div>
                </div>

                <div class="order-card-actions">
                    ${actionButtonsHtml}
                </div>
            </div>
        `;
    }).join('');
}

async function updateOrderStatus(orderId, newStatus) {
    try {
        const res = await fetch(`${API_BASE}/orders/update_status`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ order_id: orderId, status: newStatus })
        });
        const data = await res.json();
        if (res.ok && data.success) {
            showToast(data.message || `Order #${orderId} marked as ${newStatus.toUpperCase()}`, 'success');
            fetchAllData(false);
        } else {
            showToast(`⚠️ Action Blocked: ${data.detail || 'State transition forbidden.'}`, 'warning');
        }
    } catch (err) {
        showToast(`Error updating order status: ${err.message}`, 'error');
    }
}

/* ==========================================================================
   RENDER: FLOOR PLAN & TABLES
   ========================================================================== */
function renderTables() {
    let filtered = appState.tables;
    if (appState.tableSectionFilter !== 'all') {
        filtered = appState.tables.filter(t => t.section === appState.tableSectionFilter);
    }

    if (!filtered || filtered.length === 0) {
        dom.tablesGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🪑</div>
                <h3>No Tables in Section</h3>
                <p>No tables configured for section "${appState.tableSectionFilter}".</p>
            </div>
        `;
        return;
    }

    dom.tablesGrid.innerHTML = filtered.map(t => {
        const statusClass = (t.status || 'available').toLowerCase();
        const isAvail = statusClass === 'available';
        const isOcc = statusClass === 'occupied';
        const isRes = statusClass === 'reserved';

        const isOccupiedOrReserved = isOcc || isRes;

        return `
            <div class="table-card status-${statusClass}" data-table-id="${t.id}">
                <div class="table-card-top">
                    <span class="table-number">${t.table_number}</span>
                    <span class="table-status-pill status-${statusClass}">${statusClass.toUpperCase()}</span>
                </div>

                <div class="table-details">
                    <p class="table-capacity">👥 Capacity: <strong>${t.capacity} Persons</strong></p>
                    <p class="table-section">📍 Section: <strong>${t.section}</strong></p>
                </div>

                <div class="table-actions">
                    <button class="table-btn-status ${isAvail ? 'active' : ''}" onclick="updateTableStatus('${t.table_number}', 'available')">Available</button>
                    <button class="table-btn-status ${isOcc ? 'active' : ''}" onclick="updateTableStatus('${t.table_number}', 'occupied')">Occupied</button>
                    <button class="table-btn-status ${isRes ? 'active' : ''}" onclick="updateTableStatus('${t.table_number}', 'reserved')">Reserved</button>
                </div>

                ${isOccupiedOrReserved ? `
                    <button class="btn-settle-bill" onclick="openTableBillModal('${t.table_number}')">
                        🧾 Generate Bill & Settle
                    </button>
                ` : ''}
            </div>
        `;
    }).join('');
}

async function openTableBillModal(tableNumber) {
    const modal = document.getElementById('billModal');
    const title = document.getElementById('billModalTableTitle');
    const body = document.getElementById('billModalBody');
    const footer = document.getElementById('billModalFooter');

    title.textContent = `Table ${tableNumber} • Consolidated Tax Invoice`;
    body.innerHTML = `<div class="loading-spinner">Fetching active orders for Table ${tableNumber}...</div>`;
    modal.classList.add('active');

    try {
        const res = await fetch(`${API_BASE}/tables/${tableNumber}/bill`);
        const data = await res.json();

        if (!data.success || !data.has_active_orders) {
            body.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🧾</div>
                    <h3>No Active Orders</h3>
                    <p>${data.message || `Table ${tableNumber} does not have any active unpaid orders.`}</p>
                </div>
            `;
            footer.innerHTML = `<button class="btn btn-secondary" onclick="closeBillModal()">Close</button>`;
            return;
        }

        const kotsHtml = (data.kots || []).map(k => `<strong>#${k}</strong>`).join(', ');
        const itemsHtml = (data.items || []).map(it => `
            <div class="bill-item-row">
                <span><strong>${it.name}</strong> × ${it.quantity}</span>
                <span>₹${it.total_price.toFixed(2)}</span>
            </div>
        `).join('');

        body.innerHTML = `
            <div class="bill-invoice-box">
                <div class="bill-kots-meta">
                    <span>📋 Running KOTs: ${kotsHtml}</span>
                </div>
                <div class="bill-items-list">
                    ${itemsHtml}
                </div>
                <div class="bill-summary-row">
                    <span>Food Subtotal</span>
                    <span>₹${data.subtotal.toFixed(2)}</span>
                </div>
                <div class="bill-summary-row">
                    <span>GST (5%)</span>
                    <span>₹${data.tax.toFixed(2)}</span>
                </div>
                <div class="bill-grand-total">
                    <span>Grand Total Payable</span>
                    <span>₹${data.net_total.toFixed(2)}</span>
                </div>
            </div>

            <div style="margin-top: 10px;">
                <label style="font-size: 12px; color: var(--text-muted); font-weight: 600;">Select Payment Mode:</label>
                <div class="payment-mode-selector" id="paymentModeSelector">
                    <button class="payment-mode-btn active" data-mode="upi" onclick="selectPaymentMode(this)">📱 UPI</button>
                    <button class="payment-mode-btn" data-mode="cash" onclick="selectPaymentMode(this)">💵 Cash</button>
                    <button class="payment-mode-btn" data-mode="card" onclick="selectPaymentMode(this)">💳 Card</button>
                </div>
            </div>

            <!-- Dynamic UPI QR Code Card -->
            <div id="upiQrPreviewBox" style="display: block; text-align: center; background: rgba(0,0,0,0.35); border: 1px solid var(--border-highlight); border-radius: 8px; padding: 14px; margin-top: 14px;">
                <p style="font-size: 13px; font-weight: 700; color: var(--gold-light); margin-bottom: 8px;">📱 Scan to Pay ₹${data.net_total.toFixed(2)} (UPI)</p>
                <img src="${data.upi_qr_url}" alt="UPI QR Code" style="width: 160px; height: 160px; border-radius: 8px; border: 2px solid var(--border-glass); background: #fff; padding: 6px;" />
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">UPI ID: <strong style="color: #fff;">${data.upi_id}</strong> (Umaid Haveli)</p>
            </div>
        `;

        footer.innerHTML = `
            <button class="btn btn-secondary" onclick="closeBillModal()">Cancel</button>
            <button class="btn btn-primary" onclick="confirmSettleBill('${tableNumber}')">
                ✅ Settle & Free Table (₹${data.net_total.toFixed(2)})
            </button>
        `;
    } catch (err) {
        body.innerHTML = `<p style="color: #ef4444;">Failed to load bill: ${err.message}</p>`;
        footer.innerHTML = `<button class="btn btn-secondary" onclick="closeBillModal()">Close</button>`;
    }
}

let currentSelectedPaymentMode = 'upi';

function selectPaymentMode(btn) {
    document.querySelectorAll('#paymentModeSelector .payment-mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSelectedPaymentMode = btn.getAttribute('data-mode') || 'upi';

    const qrBox = document.getElementById('upiQrPreviewBox');
    if (qrBox) {
        qrBox.style.display = currentSelectedPaymentMode === 'upi' ? 'block' : 'none';
    }
}

function closeBillModal() {
    const modal = document.getElementById('billModal');
    if (modal) modal.classList.remove('active');
}

async function confirmSettleBill(tableNumber) {
    try {
        const res = await fetch(`${API_BASE}/tables/settle_bill`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                table_number: tableNumber,
                payment_mode: currentSelectedPaymentMode
            })
        });
        const data = await res.json();
        if (data.success) {
            closeBillModal();
            showToast(`Table ${tableNumber} bill settled via ${currentSelectedPaymentMode.toUpperCase()}! Table is now Available.`, 'success');
            fetchAllData(false);
        } else {
            showToast(`Error settling bill: ${data.detail || 'Failed'}`, 'error');
        }
    } catch (err) {
        showToast(`Failed to settle bill: ${err.message}`, 'error');
    }
}

async function updateTableStatus(tableNumber, status) {
    try {
        const res = await fetch(`${API_BASE}/tables/update_status`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ table_number: tableNumber, status: status })
        });
        const data = await res.json();
        if (data.success) {
            showToast(`Table ${tableNumber} updated to ${status.toUpperCase()}! AI agent synced in real-time.`, 'success');
            fetchAllData(false);
        } else {
            showToast(`Failed to update table: ${data.detail || 'Error'}`, 'error');
        }
    } catch (err) {
        showToast(`Error updating table: ${err.message}`, 'error');
    }
}

/* ==========================================================================
   RENDER: MENU & INVENTORY
   ========================================================================== */
function renderInventory() {
    let filtered = appState.inventory;
    if (appState.inventorySearchQuery) {
        filtered = appState.inventory.filter(item => 
            (item.name || '').toLowerCase().includes(appState.inventorySearchQuery) ||
            (item.category || '').toLowerCase().includes(appState.inventorySearchQuery)
        );
    }

    if (!filtered || filtered.length === 0) {
        dom.inventoryTableBody.innerHTML = `
            <tr><td colspan="7" class="text-center">No inventory items matching query.</td></tr>
        `;
        return;
    }

    dom.inventoryTableBody.innerHTML = filtered.map(item => {
        const isLow = item.stock <= item.reorder_level;
        const isAvail = item.available === 1;

        return `
            <tr class="${!isAvail ? 'row-unavailable' : ''} ${isLow ? 'row-low-stock' : ''}">
                <td><strong>${item.name}</strong></td>
                <td><span class="category-tag">${item.category || 'General'}</span></td>
                <td>₹${item.price || 0}</td>
                <td>
                    <div class="stock-adjust-cell">
                        <button class="btn-stock-delta" onclick="adjustStock(${item.item_id}, ${item.stock - 1})">-</button>
                        <span class="stock-val ${isLow ? 'text-amber' : ''}">${item.stock} ${item.unit || ''}</span>
                        <button class="btn-stock-delta" onclick="adjustStock(${item.item_id}, ${item.stock + 1})">+</button>
                    </div>
                </td>
                <td>${item.reorder_level} ${item.unit || ''}</td>
                <td>
                    <button class="status-toggle-btn ${isAvail ? 'available' : 'unavailable'}" onclick="toggleItemAvailability(${item.item_id}, ${isAvail ? 0 : 1})">
                        ${isAvail ? '🟢 In Stock' : '🔴 Out of Stock'}
                    </button>
                </td>
                <td>
                    <button class="btn-quick-order" onclick="triggerChatOrder('${item.name}')">💬 Ask AI to Order</button>
                </td>
            </tr>
        `;
    }).join('');
}

async function adjustStock(itemId, newStock) {
    if (newStock < 0) return;
    try {
        const isAvail = newStock > 0 ? 1 : 0;
        const res = await fetch(`${API_BASE}/inventory/update_stock`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_id: itemId, stock: newStock, available: isAvail })
        });
        const data = await res.json();
        if (data.success) {
            fetchAllData(false);
        }
    } catch (err) {
        showToast(`Failed to update stock: ${err.message}`, 'error');
    }
}

async function toggleItemAvailability(itemId, newAvailability) {
    try {
        const item = appState.inventory.find(i => i.item_id === itemId);
        const res = await fetch(`${API_BASE}/inventory/update_stock`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_id: itemId, available: newAvailability })
        });
        const data = await res.json();
        if (data.success) {
            showToast(`${item ? item.name : 'Item'} status changed to ${newAvailability ? 'Available' : 'Out of Stock'}! AI agent synced.`, 'info');
            fetchAllData(false);
        }
    } catch (err) {
        showToast(`Failed to update availability: ${err.message}`, 'error');
    }
}

function triggerChatOrder(itemName) {
    switchTab('chatTab');
    dom.chatInput.value = `1 ${itemName} order kardo`;
    handleSendMessage();
}

/* ==========================================================================
   AI AGENT INTERACTIVE CONSOLE & TELEMETRY
   ========================================================================== */
async function loginCustomer(phone) {
    try {
        const cleanPhone = (phone || '').trim() || '9660888489';
        const res = await fetch(`${API_BASE}/customer/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phone: cleanPhone })
        });
        const data = await res.json();
        if (data.success) {
            appState.customer = data.customer;
            appState.customerPhone = data.phone;
            
            // Update UI badge
            if (dom.customerProfileBadge) {
                const badgeText = data.customer.vip_status === 'gold' ? '⭐ Gold' : (data.customer.vip_status === 'platinum' ? '💎 Platinum' : '👤 Regular');
                dom.customerProfileBadge.textContent = `👤 ${data.customer.name} (${badgeText})`;
            }

            // Restore chat history in UI if available
            if (data.chat_history && data.chat_history.length > 0) {
                dom.chatMessages.innerHTML = '';
                data.chat_history.forEach(msg => {
                    appendChatMessage(msg.role, msg.content);
                });
            }

            if (data.active_table) {
                showToast(`Session restored for ${data.customer.name}! Active Table: ${data.active_table}`, 'info');
            } else {
                showToast(`Logged in as ${data.customer.name}`, 'success');
            }
        }
    } catch (err) {
        console.error('Customer login error:', err);
    }
}

async function handleSendMessage() {
    const text = dom.chatInput.value.trim();
    if (!text || appState.isChatting) return;

    // Append user message
    appendChatMessage('user', text);
    dom.chatInput.value = '';
    appState.isChatting = true;
    dom.btnSendMessage.disabled = true;

    // Append typing indicator
    const typingId = appendTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                phone: appState.customerPhone || '9660888489',
                session_id: `phone_${appState.customerPhone || '9660888489'}`
            })
        });

        removeTypingIndicator(typingId);
        const data = await res.json();

        if (data.success) {
            appendChatMessage('assistant', data.response);
            updateTelemetry(data);
            
            // Immediate sync if DB was modified (e.g. order placed or table booked)
            fetchAllData(false);
        } else {
            appendChatMessage('assistant', `⚠️ Error: ${data.detail || 'Server encountered an error.'}`);
        }
    } catch (err) {
        removeTypingIndicator(typingId);
        appendChatMessage('assistant', `⚠️ Connection error: Could not reach backend agent (${err.message}).`);
    } finally {
        appState.isChatting = false;
        dom.btnSendMessage.disabled = false;
        dom.chatInput.focus();
    }
}

function appendChatMessage(sender, content) {
    const isUser = sender === 'user';
    const msgDiv = document.createElement('div');
    msgDiv.className = `message-bubble ${isUser ? 'user' : 'assistant'}`;

    // Format markdown images ![alt](url)
    let formattedContent = (content || '')
        .replace(/!\[(.*?)\]\((https?:\/\/[^\s\)]+)\)/g, '<div style="margin: 12px 0; text-align: center;"><img src="$2" alt="$1" style="width: 170px; height: 170px; border-radius: 8px; border: 2px solid rgba(245,158,11,0.4); background: #fff; padding: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.4);" /><br><span style="font-size: 11px; color: var(--gold-light);">$1</span></div>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n- /g, '<br>• ')
        .replace(/\n/g, '<br>');

    msgDiv.innerHTML = `
        <div class="msg-avatar">${isUser ? '👤' : '🤖'}</div>
        <div class="msg-content">
            <p>${formattedContent}</p>
        </div>
    `;

    dom.chatMessages.appendChild(msgDiv);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

function appendTypingIndicator() {
    const id = 'typing_' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message-bubble assistant typing';
    div.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-content">
            <span class="dot-flashing"></span>
        </div>
    `;
    dom.chatMessages.appendChild(div);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function updateTelemetry(data) {
    if (!data) return;

    dom.telIntent.textContent = data.intent || 'direct_query';
    
    const tokens = data.tokens || {};
    dom.telPromptTok.textContent = tokens.prompt_tokens ?? tokens.prompt ?? 0;
    dom.telCompTok.textContent = tokens.completion_tokens ?? tokens.completion ?? 0;
    dom.telTotalTok.textContent = tokens.total_tokens ?? tokens.total ?? 0;

    // Active slots display
    if (data.active_slots) {
        dom.telActiveSlots.textContent = JSON.stringify(data.active_slots, null, 2);
    } else {
        dom.telActiveSlots.textContent = '{}';
    }

    // Dynamic SQL
    if (data.sql_query) {
        dom.telExecutedSql.textContent = Array.isArray(data.sql_query) ? data.sql_query.join('\n\n') : data.sql_query;
    } else {
        dom.telExecutedSql.textContent = '-- No SQL executed for this intent';
    }
}

/* ==========================================================================
   NOTIFICATION TOASTS
   ========================================================================== */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';
    if (type === 'warning') icon = '⚠️';

    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${message}</span>
    `;

    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// Window global bindings for inline onclicks
window.updateOrderStatus = updateOrderStatus;
window.updateTableStatus = updateTableStatus;
window.adjustStock = adjustStock;
window.toggleItemAvailability = toggleItemAvailability;
window.triggerChatOrder = triggerChatOrder;
window.openTableBillModal = openTableBillModal;
window.selectPaymentMode = selectPaymentMode;
window.closeBillModal = closeBillModal;
window.confirmSettleBill = confirmSettleBill;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', initApp);
