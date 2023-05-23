package group.vlp.web;

/**
 * Project data type.
 * <p/>
 * @author phuonglh, May 22, 2023 11:10:49 PM
 * 
 */
public class Project {
	private String title;
	private String fund;
	private String duration;
	private String role;
	
	public Project(String title, String fund, String duration, String role) {
		this.title = title;
		this.fund = fund;
		this.duration = duration;
		this.role = role;
	}
	
	public String getDuration() {
		return duration;
	}
	
	public String getTitle() {
		return title;
	}
	
	public String getFund() {
		return fund;
	}
	
	public String getRole() {
		return role;
	}
}
