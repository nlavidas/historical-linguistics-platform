# 🔧 QUICK DNS SETUP GUIDE
## Make corpus-platform.nlavid.as Work

### **OVH Control Panel Steps:**

1. **Login** to https://www.ovh.com/manager/
2. **Go to** "Domains" section
3. **Click** on `corpus-platform.nlavid.as`
4. **Go to** "DNS Zone" tab
5. **Click** "Add an entry"

### **Add A Record:**
```
Type: A
Subdomain: (leave empty)
Target: 135.125.216.3
TTL: 3600
```

6. **Click** "Add record"
7. **Wait** 5-30 minutes for DNS to propagate

### **Test After Setup:**
- Visit: https://corpus-platform.nlavid.as
- Should show secure login page
- Password: `historical_linguistics_2025`

### **If DNS Doesn't Work:**
- Try clearing browser DNS cache
- Use `nslookup corpus-platform.nlavid.as` to check
- Contact OVH support if needed

---

**🎯 Result**: https://corpus-platform.nlavid.as will work with SSL!
